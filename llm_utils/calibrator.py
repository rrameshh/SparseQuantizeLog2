# llm_utils/calibrator.py
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from llm_quantizers.llm_linear import TokenAwareQuantLinear
from llm_quantizers.llm_attention import KVCacheQuantizer

class LLMQuantCalibrator:
    """
    Calibration utility specifically designed for LLM quantization.
    """
    def __init__(self, model, calib_loader):
        self.model = model
        self.calib_loader = calib_loader
        
    def single_input_forward_hook(self, module, inp, outp):
        if module.tmp_input is None:
            module.tmp_input = []
        module.tmp_input.append(inp[0].cpu().detach())
        
    def outp_forward_hook(self, module, inp, outp):
        if module.tmp_out is None:
            module.tmp_out = []
        module.tmp_out.append(outp.cpu().detach())
        
    def kv_cache_forward_hook(self, module, inp, outp):
        """Special hook for capturing KV pairs for cache calibration."""
        if not hasattr(module, 'k_samples'):
            module.k_samples = []
            module.v_samples = []
            
        # Extract key and value tensors from model-specific outputs
        # This will need to be adapted based on the specific LLM architecture
        if isinstance(outp, tuple) and len(outp) >= 2:
            # Typical case where attention returns (attn_output, attn_weights, ...)
            k, v = inp[1], inp[2]  # Usually the second and third inputs are K and V
        else:
            # Try to extract from module directly if available
            k = getattr(module, 'k', None)
            v = getattr(module, 'v', None)
            
        if k is not None and v is not None:
            module.k_samples.append(k.cpu().detach())
            module.v_samples.append(v.cpu().detach())

    def llm_quant_calib(self):
        """
        LLM-specific calibration procedure.
        Handles token-aware quantization and KV cache optimization.
        """
        device = next(self.model.parameters()).device
        total = sum(1 for name, module in self.model.named_modules() 
                   if hasattr(module, 'calibrated') and not module.calibrated)
                   
        with tqdm(total=total) as progress_bar:
            for name, module in self.model.named_modules():
                if not hasattr(module, 'calibrated') or module.calibrated:
                    continue
                    
                progress_bar.set_description(f"calibrating {name}")
                hooks = []
                
                # Register appropriate hooks based on module type
                if isinstance(module, TokenAwareQuantLinear):
                    hooks.append(module.register_forward_hook(self.outp_forward_hook))
                    hooks.append(module.register_forward_hook(self.single_input_forward_hook))
                elif isinstance(module, KVCacheQuantizer):
                    hooks.append(module.register_forward_hook(self.kv_cache_forward_hook))
                    
                # Run calibration data through the model
                with torch.no_grad():
                    for i, (inp, attn_mask) in enumerate(self.calib_loader):
                        inp = inp.to(device)
                        if attn_mask is not None:
                            attn_mask = attn_mask.to(device)
                            _ = self.model(inp, attention_mask=attn_mask)
                        else:
                            _ = self.model(inp)
                            
                # Process collected calibration data
                if isinstance(module, TokenAwareQuantLinear):
                    module.raw_out = torch.cat(module.tmp_out, dim=0)
                    module.raw_input = torch.cat(module.tmp_input, dim=0)
                    module.tmp_input = module.tmp_out = None
                    
                    # Run hyperparameter search
                    with torch.no_grad():
                        module.hyperparameter_searching()
                        
                elif isinstance(module, KVCacheQuantizer):
                    # Prepare and calibrate KV cache quantizer
                    k_samples = torch.cat(module.k_samples, dim=0)
                    v_samples = torch.cat(module.v_samples, dim=0)
                    module.calibrate(k_samples, v_samples)
                    delattr(module, 'k_samples')
                    delattr(module, 'v_samples')
                    
                # Remove hooks
                for hook in hooks:
                    hook.remove()
                    
                progress_bar.update()
                
        # Set all modules to quantized forward mode
        for name, module in self.model.named_modules():
            if hasattr(module, 'mode'):
                module.mode = "quant_forward"

class TinyBertQuantCalibrator(LLMQuantCalibrator):
    def __init__(self, model, calib_loader):
        super().__init__(model, calib_loader)
        
    def tinybert_forward_hook(self, module, inp, outp):
        """Special hook for TinyBERT attention layers"""
        if not hasattr(module, 'attention_scores'):
            return
            
        # TinyBERT stores attention differently
        k = module.key_layer(inp[0])
        v = module.value_layer(inp[0])
        
        if not hasattr(module, 'k_samples'):
            module.k_samples = []
            module.v_samples = []
            
        module.k_samples.append(k.detach().cpu())
        module.v_samples.append(v.detach().cpu())

    def llm_quant_calib(self):
        # Add TinyBERT-specific hooks before standard calibration
        for name, module in self.model.named_modules():
            if 'attention' in name and hasattr(module, 'key_layer'):
                module.register_forward_hook(self.tinybert_forward_hook)
                
        # Run standard calibration
        super().llm_quant_calib()