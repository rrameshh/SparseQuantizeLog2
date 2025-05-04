from transformers import AutoModelForSequenceClassification
import torch.nn as nn
import torch
from llm_quantizers.llm_linear import TokenAwareQuantLinear


def quantize_tinybert(model, w_bit=4, a_bit=4, kv_bit=3):
    """Quantize TinyBERT model using custom quantization layers"""
    # Get model config
    config = model.config
    
    # Replace linear layers recursively
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Skip classifier layer to maintain precision
            if name == 'classifier':
                continue
                
            # Create quantized linear layer
            quant_linear = TokenAwareQuantLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                w_bit=w_bit,
                a_bit=a_bit,
                max_position=config.max_position_embeddings
            )
            
            # Copy original weights
            quant_linear.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                quant_linear.bias.data.copy_(module.bias.bias)
                
            setattr(model, name, quant_linear)
            
        elif isinstance(module, nn.ModuleList):  # Handle layer lists
            for i, layer in enumerate(module):
                module[i] = quantize_tinybert(layer, w_bit, a_bit, kv_bit)
    
    return model