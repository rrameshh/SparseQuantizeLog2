# tubgemm_utils/model_converter.py
import torch
import torch.nn as nn
from tubgemm_quantizers.tublog_linear import TubLogLinear
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def convert_model_to_tublog(model, w_bit=4, a_bit=4, sparsity_threshold=0.01, progressive=True, activation_aware=True):
    """
    Convert a transformer model to use TubLog quantization with progressive depth
    
    Args:
        model: The model to convert
        w_bit: Bit width for weights
        a_bit: Bit width for activations
        sparsity_threshold: Threshold for sparsification
        progressive: Whether to use progressive sparsity (deeper = more sparse)
        activation_aware: Whether to enable activation-aware quantization
    """
    # Track modules that need replacement
    replacements = {}
    
    # Count layers to set progressive sparsity
    total_layers = sum(1 for name, module in model.named_modules() 
                       if isinstance(module, nn.Linear))
    layer_count = 0
    
    # Find all linear layers that should be replaced
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.rsplit('.', 1)[1] if '.' in name else name
            
            # Skip certain layers if needed (e.g. output classifier)
            if 'classifier' in name:
                print(f"Skipping quantization for {name}")
                continue
            
            layer_count += 1
            
            # Set progressive sparsity (lower in earlier layers)
            if progressive:
                layer_pos = layer_count / total_layers  # 0 to 1
                layer_sparsity = sparsity_threshold * (0.5 + 0.5 * layer_pos)
                print(f"Quantizing: {name} with sparsity {layer_sparsity:.4f}")
            else:
                layer_sparsity = sparsity_threshold
                print(f"Quantizing: {name}")
                
            # Create quantized replacement layer
            tublog_layer = TubLogLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                w_bit=w_bit,
                a_bit=a_bit,
                sparsity_threshold=layer_sparsity,
                mode="raw"  # Start in raw mode for calibration
            )
            
            # Copy weights and biases
            tublog_layer.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                tublog_layer.bias.data.copy_(module.bias.data)
            
            # Enable or disable activation-aware quantization
            tublog_layer.w_quantizer.activation_aware = activation_aware
            tublog_layer.a_quantizer.activation_aware = activation_aware
            
            # Store replacement information
            replacements[name] = (parent_name, child_name, module, tublog_layer)
    
    # Apply replacements
    for name, (parent_name, child_name, _, tublog_layer) in replacements.items():
        if parent_name:
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, tublog_layer)
        else:
            setattr(model, child_name, tublog_layer)
    
    return model

def calibrate_tublog_model(model, dataloader, device='cpu', is_encoder_decoder=False, max_batches=5):
    """
    Calibrate a TubLog-quantized model using a dataloader
    
    Args:
        model: The model to calibrate
        dataloader: DataLoader for calibration data
        device: Device to run calibration on
        is_encoder_decoder: Whether the model is an encoder-decoder model
        max_batches: Maximum number of batches to use for calibration
    """
    model.eval()
    model.to(device)

    print("\n=== Calibration Debug ===")
    print(f"Using device: {device}")

    # First pass: collect activation statistics
    try:
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc="Collecting calibration data")):
                if i >= max_batches:
                    break
                    
                print(f"\nBatch {i} shape: {batch.shape if isinstance(batch, torch.Tensor) else [t.shape for t in batch if isinstance(t, torch.Tensor)]}")

                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                else:
                    inputs = batch.to(device)

                print(f"Input to model shape: {inputs.shape}")

                try:
                    if is_encoder_decoder:
                        decoder_input_ids = torch.zeros((inputs.shape[0], 1), dtype=torch.long, device=device)
                        model(
                            input_ids=inputs,
                            decoder_input_ids=decoder_input_ids
                        )
                    else:
                        model(input_ids=inputs)  # Forward pass for BERT-style models
                except Exception as e:
                    print(f"Error during forward pass: {str(e)}")
                    # Continue to the next batch instead of raising
                    continue
    except Exception as e:
        print(f"Error during activation statistics collection: {str(e)}")
        # Continue to calibration
                    
    # After collecting activation statistics, run explicit calibration on each layer
    print("\n=== Performing Explicit Calibration ===")
    for name, module in model.named_modules():
        if isinstance(module, TubLogLinear):
            try:
                # Check if module needs calibration
                needs_calibration = not hasattr(module, 'calibrated') or not module.calibrated
                
                if needs_calibration:
                    print(f"Calibrating {name}...")
                    
                    # Create dummy input with appropriate shape
                    try:
                        dummy_input = torch.randn(2, module.in_features, device=device)
                        module.calibrate(dummy_input)
                    except Exception as e:
                        print(f"Error calibrating {name} with dummy input: {str(e)}")
                        # Try with a different shaped input
                        try:
                            # Try a 1D input
                            dummy_input = torch.randn(module.in_features, device=device)
                            module.calibrate(dummy_input)
                        except Exception as e2:
                            print(f"Error with fallback calibration for {name}: {str(e2)}")
            except Exception as e:
                print(f"Error processing module {name}: {str(e)}")
                
    print("\n=== Calibration Complete ===")

def print_q_parameter_values(model):
    """
    Print q parameter values for all TubLogQuantizer instances in the model
    
    Args:
        model: The model with TubLogQuantizer instances
    
    Returns:
        Dictionary of q values
    """
    q_values = {}
    w_q_values = []
    a_q_values = []
    
    for name, module in model.named_modules():
        if hasattr(module, 'q'):
            q_values[name] = module.q.item()
        elif hasattr(module, 'w_quantizer') and hasattr(module.w_quantizer, 'q'):
            q_values[f"{name}.w_quantizer"] = module.w_quantizer.q.item()
            w_q_values.append(module.w_quantizer.q.item())
        elif hasattr(module, 'a_quantizer') and hasattr(module.a_quantizer, 'q'):
            q_values[f"{name}.a_quantizer"] = module.a_quantizer.q.item()
            a_q_values.append(module.a_quantizer.q.item())
    
    print("\n=== q Parameter Values ===")
    for name, q_val in q_values.items():
        print(f"{name}: {q_val:.4f}")
    
    # Print statistics
    if w_q_values:
        print(f"\n=== Weight q Statistics ===")
        print(f"Mean: {np.mean(w_q_values):.4f}")
        print(f"Min: {np.min(w_q_values):.4f}")
        print(f"Max: {np.max(w_q_values):.4f}")
        print(f"Std Dev: {np.std(w_q_values):.4f}")
    
    if a_q_values:
        print(f"\n=== Activation q Statistics ===")
        print(f"Mean: {np.mean(a_q_values):.4f}")
        print(f"Min: {np.min(a_q_values):.4f}")
        print(f"Max: {np.max(a_q_values):.4f}")
        print(f"Std Dev: {np.std(a_q_values):.4f}")
    
    # Optional: Create histograms to visualize distributions
    if len(w_q_values) > 5 or len(a_q_values) > 5:
        try:
            plt.figure(figsize=(12, 6))
            
            if w_q_values:
                plt.subplot(1, 2, 1)
                plt.hist(w_q_values, bins=min(20, len(w_q_values)))
                plt.title('Distribution of Weight q Values')
                plt.xlabel('q Value')
                plt.ylabel('Count')
            
            if a_q_values:
                plt.subplot(1, 2, 2)
                plt.hist(a_q_values, bins=min(20, len(a_q_values)))
                plt.title('Distribution of Activation q Values')
                plt.xlabel('q Value')
                plt.ylabel('Count')
            
            plt.tight_layout()
            plt.savefig('q_parameter_distribution.png')
            print(f"Histogram saved to q_parameter_distribution.png")
        except ImportError:
            print("Matplotlib not available for histogram visualization")
    
    return q_values

def print_tublog_statistics(model):
    """
    Print statistics about the quantized model
    
    Args:
        model: The TubLog-quantized model
    """
    total_params = 0
    quantized_params = 0
    total_weight_sparsity = 0
    total_layers = 0
    
    # For collecting q values
    w_q_values = []
    a_q_values = []
    
    # Collect statistics
    for name, module in model.named_modules():
        if isinstance(module, TubLogLinear):
            total_layers += 1
            
            # Count parameters
            param_count = module.weight.numel()
            total_params += param_count
            quantized_params += param_count
            
            # Calculate sparsity
            if module.calibrated:
                w_sim = module.w_quantizer(module.weight)
                sparsity = (w_sim == 0).sum().item() / w_sim.numel()
                total_weight_sparsity += sparsity
                
                print(f"Layer: {name}")
                print(f"  Weight shape: {module.weight.shape}")
                print(f"  Sparsity: {sparsity*100:.2f}%")
                
                # Scale information
                if module.w_quantizer.scale.numel() == 1:
                    print(f"  Weight scale: {module.w_quantizer.scale.item():.6f}")
                else:
                    print(f"  Weight scale: {module.w_quantizer.scale.mean():.6f} (mean of {module.w_quantizer.scale.numel()} values)")
                
                # q parameter values
                w_q = module.w_quantizer.q.item()
                a_q = module.a_quantizer.q.item()
                w_q_values.append(w_q)
                a_q_values.append(a_q)
                print(f"  Weight q: {w_q:.6f}")
                print(f"  Activation q: {a_q:.6f}")
                
                # Print activation statistics if available
                if hasattr(module, 'act_stats_mean'):
                    print(f"  Activation mean: {module.act_stats_mean.item():.6f}")
                    print(f"  Activation variance: {module.act_stats_var.item():.6f}")
                
                # Print first few original vs quantized weights
                orig = module.weight.data.flatten()[:3].cpu().detach().numpy()
                quant = w_sim.flatten()[:3].cpu().detach().numpy()
                print(f"  Original weights (first 3): {orig}")
                print(f"  Quantized weights (first 3): {quant}")
                print()
    
    # Print summary
    if total_layers > 0:
        avg_sparsity = total_weight_sparsity / total_layers
        avg_w_q = np.mean(w_q_values) if w_q_values else 0
        avg_a_q = np.mean(a_q_values) if a_q_values else 0
        
        print(f"\nModel statistics:")
        print(f"  Total layers quantized: {total_layers}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Quantized parameters: {quantized_params:,} ({quantized_params/total_params*100:.2f}%)")
        print(f"  Average weight sparsity: {avg_sparsity*100:.2f}%")
        print(f"  Average weight q: {avg_w_q:.4f}")
        print(f"  Average activation q: {avg_a_q:.4f}")
        
        # Visualize q distribution
        if len(w_q_values) > 5:
            try:
                plt.figure(figsize=(12, 6))
                
                plt.subplot(1, 2, 1)
                plt.hist(w_q_values, bins=min(20, len(w_q_values)))
                plt.title('Distribution of Weight q Values')
                plt.xlabel('q Value')
                plt.ylabel('Count')
                
                if a_q_values:
                    plt.subplot(1, 2, 2)
                    plt.hist(a_q_values, bins=min(20, len(a_q_values)))
                    plt.title('Distribution of Activation q Values')
                    plt.xlabel('q Value')
                    plt.ylabel('Count')
                
                plt.tight_layout()
                plt.savefig('model_q_distribution.png')
                print(f"q distribution saved to model_q_distribution.png")
            except Exception as e:
                print(f"Error creating visualization: {str(e)}")
    
    return {
        'total_layers': total_layers,
        'total_params': total_params,
        'avg_sparsity': avg_sparsity if total_layers > 0 else 0,
        'avg_weight_q': avg_w_q,
        'avg_activation_q': avg_a_q
    }