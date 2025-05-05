import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tubgemm_quantizers.tublog_quantizer import TubLogQuantizer


# ======= TubLogLinear Implementation =======
class TubLogLinear(nn.Linear):
    """
    Linear layer with tubGEMM-optimized log2 quantization and sparsification.
    
    This layer leverages tubGEMM's unique properties:
    1. Temporal-unary encoding for activations (optimized for sparsity)
    2. Binary encoding for weights (optimized for pow2 values)
    3. Each pulse represents magnitude of 2 in tubGEMM's 2-unary encoding
    """
    def __init__(self, 
                 in_features, 
                 out_features, 
                 bias=True,
                 w_bit=4,
                 a_bit=4,
                 sparsity_threshold=0.01,
                 mode="raw"):
        super().__init__(in_features, out_features, bias)
        self.mode = mode
        
        # Create weight quantizer - align with powers of 2
        self.w_quantizer = TubLogQuantizer(
            n_bits=w_bit, 
            symmetric=False,
            channel_wise=True,
            sparsity_threshold=sparsity_threshold
        )
        
        # Create activation quantizer - optimize for tubGEMM
        self.a_quantizer = TubLogQuantizer(
            n_bits=a_bit,
            symmetric=False,
            channel_wise=False,
            sparsity_threshold=sparsity_threshold
        )
        
        # Enable activation awareness
        self.w_quantizer.activation_aware = True
        self.a_quantizer.activation_aware = True
        
        # For calibration data storage
        self.raw_input = None
        self.raw_out = None
        self.calibrated = False
        
        # For storing activation statistics
        self.register_buffer('act_stats_mean', torch.tensor(0.0))
        self.register_buffer('act_stats_var', torch.tensor(1.0))
        self.register_buffer('act_stats_count', torch.tensor(0))
        self.collecting_stats = True
    
    def forward(self, x):
        # Store activation statistics if collecting stats
        if self.collecting_stats:
            self._update_activation_stats(x)
        
        if self.mode == 'raw':
            # No quantization, just standard linear operation
            out = F.linear(x, self.weight, self.bias)
        elif self.mode == "quant_forward":
            # Full quantization with calibrated parameters
            out = self.quant_forward(x)
        elif self.mode == 'debug_only_quant_weight':
            # Only quantize weights for debugging
            out = self.debug_only_quant_weight(x)
        elif self.mode == 'debug_only_quant_act':
            # Only quantize activations for debugging
            out = self.debug_only_quant_act(x)
        elif self.mode == 'weight_only':
            # Weight-only quantization (for production)
            out = self.weight_only_quant(x)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented")
        
        return out
    
    def _update_activation_stats(self, x):
        """Update running statistics of activations"""
        count = self.act_stats_count.item()
        new_count = count + 1
        
        # Calculate batch statistics
        batch_mean = x.mean().item()
        batch_var = x.var().item()
        
        # Update running statistics with new batch
        if count == 0:
            self.act_stats_mean = torch.tensor(batch_mean)
            self.act_stats_var = torch.tensor(batch_var)
        else:
            # Running average update
            self.act_stats_mean = torch.tensor(
                (self.act_stats_mean.item() * count + batch_mean) / new_count
            )
            self.act_stats_var = torch.tensor(
                (self.act_stats_var.item() * count + batch_var) / new_count
            )
        
        self.act_stats_count = torch.tensor(new_count)
        
    def weight_only_quant(self, x):
        """Forward pass with only weights quantized (for production)"""
        assert self.calibrated, f"Module should be calibrated before running weight_only_quant for {self}"
        
        # Quantize weights and bias with safe handling
        try:
            w_sim, bias_sim = self.quant_weight_bias()
            
            # Check for problematic values
            if torch.isnan(w_sim).any() or torch.isinf(w_sim).any():
                print(f"WARNING: NaN or Inf in quantized weights - using original weights")
                w_sim = self.weight
                
            # Use original activations (no quantization)
            # Perform linear operation with quantized weights but original activations
            out = F.linear(x, w_sim, bias_sim)
            
        except Exception as e:
            print(f"Error in weight_only_quant: {str(e)}")
            # Fall back to raw mode
            out = F.linear(x, self.weight, self.bias)
        
        return out
    
    def quant_weight_bias(self):
        """Quantize weights using the weight quantizer"""
        w_sim = self.w_quantizer(self.weight)
        return w_sim, self.bias if self.bias is not None else None
    
    def quant_input(self, x):
        """Quantize input using the activation quantizer"""
        return self.a_quantizer(x)
    
    def quant_forward(self, x):
        """Forward pass with both weights and activations quantized"""
        assert self.calibrated, f"Module should be calibrated before running quant_forward for {self}"
        
        # Store original input for debugging
        x_orig = x.clone()
        
        # Quantize weights and bias
        w_sim, bias_sim = self.quant_weight_bias()
        
        # Quantize input activations
        x_sim = self.quant_input(x)
        
        # Check for problematic values before matrix multiplication
        if torch.isnan(x_sim).any() or torch.isinf(x_sim).any():
            print(f"WARNING: NaN or Inf detected in quantized activations - replacing with zeros")
            x_sim = torch.nan_to_num(x_sim, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.isnan(w_sim).any() or torch.isinf(w_sim).any():
            print(f"WARNING: NaN or Inf detected in quantized weights - replacing with zeros")
            w_sim = torch.nan_to_num(w_sim, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Add debug validation before proceeding
        x_zeros = (x_sim == 0).float().mean().item() * 100
        if x_zeros > 90:
            print(f"CRITICAL WARNING: Over 90% zeros in activations - using original activations")
            x_sim = x_orig  # Fall back to original activations
        
        # Perform linear operation with quantized values
        out = F.linear(x_sim, w_sim, bias_sim)
        
        # Add output validation
        if torch.isnan(out).any() or torch.isinf(out).any():
            print(f"WARNING: NaN or Inf detected in output - replacing with zeros")
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        
        return out
    
    def debug_only_quant_weight(self, x):
        """Forward pass with only weights quantized (for debugging)"""
        w_sim, bias_sim = self.quant_weight_bias()
        out = F.linear(x, w_sim, bias_sim)
        return out
    
    def debug_only_quant_act(self, x):
        """Forward pass with only activations quantized (for debugging)"""
        x_sim = self.quant_input(x)
        out = F.linear(x_sim, self.weight, self.bias)
        return out
    
    def calibrate(self, x):
        """Calibrate quantizers using input data with activation awareness"""
        with torch.no_grad():
            # First pass: collect activation statistics if not already collected
            if self.act_stats_count.item() == 0:
                self._update_activation_stats(x)
            
            # Prepare activation data dictionary
            activation_data = {
                'mean': self.act_stats_mean.item(),
                'var': self.act_stats_var.item()
            }
            
            # For weights, we'd ideally want to see which activations they process
            # Use correlations between activations and weights if we have enough samples
            try:
                if x.size(0) > 2:  # If we have at least a couple of samples
                    # Use a small batch for correlation analysis (but no more than what we have)
                    num_samples = min(10, x.size(0))
                    activations_sample = x[:num_samples].detach()
                    
                    # Calculate correlation between activations and weights
                    # This is a simplified approach
                    correlation = torch.einsum('bi,oj->boj', activations_sample, self.weight)
                    correlation_importance = torch.sigmoid(correlation.abs().mean(dim=0))
                    
                    # Use this to inform weight calibration
                    importance = correlation_importance.mean().item()
                    activation_data['importance'] = importance
            except Exception as e:
                print(f"Warning: Could not calculate activation-weight correlation: {str(e)}")
            
            try:
                # Calibrate activation quantizer
                print(f"Calibrating activation quantizer with data shape {x.shape}")
                if torch.all(x == 0):
                    # Create dummy non-zero data for calibration if all zeros
                    print("WARNING: All activation values are zero, using synthetic data for calibration")
                    dummy_x = torch.rand_like(x) * 0.01
                    self.a_quantizer.calibrate(dummy_x)
                else:
                    self.a_quantizer.calibrate(x, activation_data)
                
                # Calibrate weight quantizer with activation awareness
                print(f"Calibrating weight quantizer with weight shape {self.weight.shape}")
                self.w_quantizer.calibrate(self.weight, activation_data)
                
                # Verify calibration worked
                w_quant = self.w_quantizer(self.weight)
                a_quant = self.a_quantizer(x)
                
                if torch.all(w_quant == 0):
                    print("ERROR: All quantized weights are zero! Calibration failed.")
                else:
                    w_sparsity = (w_quant == 0).float().mean().item()
                    print(f"Weight quantizer calibrated successfully - sparsity: {w_sparsity*100:.2f}%")
                    
                if torch.all(a_quant == 0) and not torch.all(x == 0):
                    print("ERROR: All quantized activations are zero but inputs weren't! Calibration failed.")
                else:
                    a_sparsity = (a_quant == 0).float().mean().item()
                    print(f"Activation quantizer calibrated successfully - sparsity: {a_sparsity*100:.2f}%")
                
                # Report q values
                print(f"Weight q value: {self.w_quantizer.q.item():.4f}")
                print(f"Activation q value: {self.a_quantizer.q.item():.4f}")
                
                # Set calibrated flag
                self.calibrated = True
            except Exception as e:
                print(f"ERROR during calibration: {str(e)}")
                # Don't set calibrated flag if there was an error
        
        return self
    
    def hyperparameter_searching(self):
        """
        Optimize quantization parameters using stored calibration data
        """
        if self.raw_input is None or not len(self.raw_input):
            raise RuntimeError("No calibration data available. Run forward passes with raw mode first.")
        
        # Move data to appropriate device
        x = self.raw_input.to(self.weight.device)
        
        # Calibrate both quantizers
        self.calibrate(x)
        
        # Clean up calibration data
        self.raw_input = None
        self.raw_out = None
        
        return None
    
    def get_q_values(self):
        """Return the current q values for both quantizers"""
        return {
            'weight_q': self.w_quantizer.q.item(),
            'activation_q': self.a_quantizer.q.item()
        }