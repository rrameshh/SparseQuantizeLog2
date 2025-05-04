# llm_quantizers/llm_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from llm_quantizers.llm_logarithm import AdaLogLLMQuantizer

class KVCacheQuantizer(nn.Module):
    """
    Specialized quantizer for key-value cache in LLMs.
    
    This quantizer is designed to minimize memory footprint of the KV cache
    while maintaining accuracy during inference.
    """
    def __init__(self, n_bits=3, num_heads=12, head_dim=64):
        super().__init__()
        self.n_bits = n_bits
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inited = False
        
        # Create separate quantizers for keys and values
        self.k_quantizer = AdaLogLLMQuantizer(n_bits=n_bits, symmetric=False, channel_wise=True)
        self.v_quantizer = AdaLogLLMQuantizer(n_bits=n_bits, symmetric=False, channel_wise=True)
        
        # Reserve space for quantized cache
        self.register_buffer('quantized_k_cache', None)
        self.register_buffer('quantized_v_cache', None)
        
    def init_kv_cache(self, batch_size, max_seq_len):
        """Initialize empty KV cache with quantization-friendly format."""
        # Initialize with zeros - will be filled during inference
        # Store in quantized format directly to save memory
        self.quantized_k_cache = torch.zeros(
            batch_size, self.num_heads, max_seq_len, self.head_dim,
            dtype=torch.uint8, device='cuda')
        
        self.quantized_v_cache = torch.zeros(
            batch_size, self.num_heads, max_seq_len, self.head_dim,
            dtype=torch.uint8, device='cuda')
            
    def update_kv_cache(self, k, v, positions):
        """
        Update the KV cache with new key-value pairs at specified positions.
        
        Args:
            k: New keys [batch, num_heads, seq_len, head_dim]
            v: New values [batch, num_heads, seq_len, head_dim]
            positions: Token positions to update
        """
        batch_size, _, seq_len, _ = k.shape
        
        # Quantize new keys and values
        k_quant = self.quantize_keys(k)
        v_quant = self.quantize_values(v)
        
        # Update cache at specified positions
        for i, pos in enumerate(positions):
            if pos < self.quantized_k_cache.shape[2]:
                self.quantized_k_cache[:, :, pos, :] = k_quant[:, :, i, :]
                self.quantized_v_cache[:, :, pos, :] = v_quant[:, :, i, :]
                
    def get_kv_from_cache(self, positions=None):
        """
        Retrieve keys and values from the cache.
        
        Args:
            positions: Optional token positions to retrieve (None = all)
            
        Returns:
            Dequantized keys and values
        """
        if positions is None:
            # Return all cached keys and values
            k_dequant = self.dequantize_keys(self.quantized_k_cache)
            v_dequant = self.dequantize_values(self.quantized_v_cache)
        else:
            # Return only specified positions
            k_quant = self.quantized_k_cache[:, :, positions, :]
            v_quant = self.quantized_v_cache[:, :, positions, :]
            
            k_dequant = self.dequantize_keys(k_quant)
            v_dequant = self.dequantize_values(v_quant)
            
        return k_dequant, v_dequant
        
    def quantize_keys(self, k):
        """Quantize keys for storage in cache."""
        # Apply special key-specific quantization
        # Keys often have different distributions than values
        return self.k_quantizer(k)
        
    def quantize_values(self, v):
        """Quantize values for storage in cache."""
        return self.v_quantizer(v)
        
    def dequantize_keys(self, k_quant):
        """Dequantize keys from cache for use in attention."""
        # Special dequantization for keys
        return self.k_quantizer.dequantize(k_quant)
        
    def dequantize_values(self, v_quant):
        """Dequantize values from cache for use in attention."""
        return self.v_quantizer.dequantize(v_quant)
        
    def calibrate(self, k_samples, v_samples):
        """Calibrate the quantizers using sample KV pairs."""
        # Initialize key quantizer
        k_max = k_samples.abs().max()
        self.k_quantizer.scale = k_max / (self.k_quantizer.n_levels - 0.5)
        self.k_quantizer.cluster_activations(k_samples)
        self.k_quantizer.inited = True
        
        # Initialize value quantizer
        v_max = v_samples.abs().max()
        self.v_quantizer.scale = v_max / (self.v_quantizer.n_levels - 0.5)
        self.v_quantizer.cluster_activations(v_samples)
        self.v_quantizer.inited = True
        
        self.inited = True