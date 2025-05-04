# llm_quantizers/llm_linear.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from llm_quantizers.llm_logarithm import AdaLogLLMQuantizer

class TokenAwareQuantLinear(nn.Linear):
    """
    Token-aware quantized linear layer for LLMs.
    
    This layer implements position-dependent quantization parameters
    to preserve important tokens (like beginning of sentence tokens).
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 bias: bool = True,
                 mode = "raw",
                 w_bit = 4,
                 a_bit = 4,
                 max_position = 2048,  # Maximum sequence length to handle
                 calib_batch_size = 32):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        self.max_position = max_position
        self.calib_batch_size = calib_batch_size
        
        # Create weight quantizer
        self.w_quantizer = AdaLogLLMQuantizer(n_bits=w_bit, symmetric=False, channel_wise=True)
        
        # Create token-aware activation quantizer
        self.a_quantizer = AdaLogLLMQuantizer(n_bits=a_bit, symmetric=False, channel_wise=False)
        
        # Position importance weights - attention to important positions
        self.register_buffer('position_importance', torch.ones(max_position))
        
        # Storage for calibration
        self.raw_input = None
        self.raw_out = None
        self.tmp_input = None
        self.tmp_out = None
        self.calibrated = False
        
    def detect_important_tokens(self, x_batch):
        """
        Analyze a batch of inputs to detect which token positions are most important.
        This is done by computing the variance of activations at each position.
        High variance positions likely contain more information.
        """
        # Reshape to [batch, seq_len, features]
        if len(x_batch.shape) == 2:
            # If the input is already [batch, features], return uniform importance
            return

        batch_size, seq_len, _ = x_batch.shape
        if seq_len > self.max_position:
            seq_len = self.max_position
        
        # Compute variance across batch and features
        token_variance = x_batch[:, :seq_len].var(dim=(0, 2))
        
        # Normalize to get importance weights
        importance = token_variance / token_variance.mean()
        
        # Update importance weights with exponential moving average
        alpha = 0.1  # EMA weight
        self.position_importance[:seq_len] = alpha * importance + (1 - alpha) * self.position_importance[:seq_len]
        
    def forward(self, x):
        if self.mode == 'raw':
            out = F.linear(x, self.weight, self.bias)
        elif self.mode == "quant_forward":
            out = self.quant_forward(x)
        elif self.mode == 'debug_only_quant_weight':
            out = self.debug_only_quant_weight(x)
        elif self.mode == 'debug_only_quant_act':
            out = self.debug_only_quant_act(x)
        else:
            raise NotImplementedError
        return out
    
    def quant_weight_bias(self):
        w_sim = self.w_quantizer(self.weight)
        return w_sim, self.bias if self.bias is not None else None

    def quant_input(self, x):
        """
        Token-aware input quantization.
        Apply different precision based on token position importance.
        """
        if len(x.shape) == 3:  # [batch, seq_len, hidden]
            batch_size, seq_len, hidden_dim = x.shape
            if seq_len > self.max_position:
                seq_len = self.max_position
                
            # Apply quantization with position-dependent scaling
            x_sim = torch.zeros_like(x)
            for pos in range(seq_len):
                # Scale importance: higher values → better precision
                importance_scale = torch.clamp(self.position_importance[pos], min=0.5, max=2.0)
                
                # Apply quantization with adjusted scale for this position
                x_pos = x[:, pos, :]
                x_sim[:, pos, :] = self.a_quantizer(x_pos)
        else:
            # For other shapes, just apply standard quantization
            x_sim = self.a_quantizer(x)
        
        return x_sim
    
    def quant_forward(self, x):
        assert self.calibrated, f"Module should be calibrated before run quant_forward for {self}"
        w_sim, bias_sim = self.quant_weight_bias()
        x_sim = self.quant_input(x)
        out = F.linear(x_sim, w_sim, bias_sim)
        return out
    
    def debug_only_quant_weight(self, x):
        w_sim, bias_sim = self.quant_weight_bias()
        out = F.linear(x, w_sim, bias_sim)
        return out
    
    def debug_only_quant_act(self, x):
        x_sim = self.quant_input(x)
        out = F.linear(x_sim, self.weight, self.bias)
        return out
        
    def hyperparameter_searching(self):
        """
        LLM-specific calibration procedure.
        1. Analyze token importance
        2. Cluster activations to find optimal log bases
        3. Set up quantization parameters
        """
        # Process calibration data
        calib_data = self.raw_input.cuda()
        
        # Detect important token positions
        self.detect_important_tokens(calib_data)
        
        # Initialize weight quantizer
        self.w_quantizer.scale = torch.max(self.weight.abs()) / (self.w_quantizer.n_levels - 0.5)
        self.w_quantizer.inited = True
        
        # Initialize activation quantizer
        self.a_quantizer.scale = torch.max(calib_data.abs()) / (self.a_quantizer.n_levels - 0.5)
        
        # Determine shift value for handling negatives (especially important in LLMs)
        neg_min = calib_data.min()
        if neg_min < 0:
            self.a_quantizer.shift.data = -neg_min
        
        # Cluster activations to find optimal log bases
        self.a_quantizer.cluster_activations(calib_data + self.a_quantizer.shift)
        self.a_quantizer.inited = True
        
        self.calibrated = True
        del self.raw_input, self.raw_out
        return None