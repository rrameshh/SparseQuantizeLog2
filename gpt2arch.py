

import torch
import time
from transformers import BertForMaskedLM, BertTokenizer, AutoModelForMaskedLM, AutoTokenizer
from tubgemm_utils.model_converter import convert_model_to_tublog, print_tublog_statistics
from tqdm.auto import tqdm

def calculate_bert_perplexity(model, tokenizer, test_texts, device="cpu"):
    """Calculate perplexity for BERT models using masked language modeling loss"""
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    
    for text in tqdm(test_texts, desc="Calculating perplexity"):
        try:
            # Tokenize the input text
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get input_ids and create labels (same as input_ids for calculating loss)
            input_ids = inputs["input_ids"]
            labels = input_ids.clone()
            
            # For BERT MLM, we need to mask some tokens or provide all tokens as labels
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=inputs["attention_mask"],
                    labels=labels  # This enables loss calculation
                )
                
                loss = outputs.loss
                
                # Count tokens excluding padding and special tokens
                non_special_mask = (labels != tokenizer.pad_token_id) & (labels != tokenizer.cls_token_id) & (labels != tokenizer.sep_token_id)
                seq_length = non_special_mask.sum().item()
                
                total_loss += loss.item() * seq_length
                total_tokens += seq_length
                
        except Exception as e:
            print(f"Error processing text: {text[:50]}... - {str(e)}")
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity

class BertCalibrationDataset(torch.utils.data.Dataset):
    """Calibration dataset for BERT models"""
    def __init__(self, tokenizer, texts, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Return input_ids tensor without batch dimension
        return encoding["input_ids"].squeeze(0)

def calibrate_bert_model(model, dataloader, device='cpu'):
    """Calibrate BERT model with TubLog quantization"""
    model.eval()
    model.to(device)
    
    print("\n=== Calibration Debug ===")
    print(f"Using device: {device}")
    
    # To store sample activations for each layer for later calibration
    sample_activations = {}
    
    # First pass: Collect sample activations for each layer
    print("\nPhase 1: Collecting activations for calibration")
    
    def save_activations_hook(name):
        def hook_fn(module, input, output):
            if name not in sample_activations:
                sample_activations[name] = input[0].detach().clone()
                print(f"Saved input activations for {name}, shape: {input[0].shape}")
            return None
        return hook_fn
    
    # Register hooks to collect activations
    handles = []
    for name, module in model.named_modules():
        if hasattr(module, "w_quantizer") and hasattr(module, "a_quantizer"):
            handle = module.register_forward_hook(save_activations_hook(name))
            handles.append(handle)
    
    # Run a few samples through the model to collect activations
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Collecting activations")):
            if i >= 2:  # Only need a few batches for calibration
                break
                
            # Add batch dimension if needed
            if batch.dim() == 1:
                batch = batch.unsqueeze(0)
                
            # Move to device
            inputs = batch.to(device)
            
            # Run forward pass to trigger hooks
            attention_mask = torch.ones_like(inputs)
            model(input_ids=inputs, attention_mask=attention_mask)
    
    # Remove hooks after collecting activations
    for handle in handles:
        handle.remove()
        
    # Second pass: Calibrate weight and activation quantizers
    print("\nPhase 2: Calibrating quantizers")
    calibrated_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "w_quantizer") and hasattr(module, "a_quantizer"):
            try:
                print(f"Calibrating {name}")
                
                # Calibrate weight quantizer
                module.w_quantizer.calibrate(module.weight)
                
                # Calibrate activation quantizer
                if name in sample_activations:
                    module.a_quantizer.calibrate(sample_activations[name])
                    # Test activation quantization
                    a_quant = module.a_quantizer(sample_activations[name])
                    a_sparsity = (a_quant == 0).float().mean().item()
                    print(f"  Activation quantizer sparsity: {a_sparsity*100:.2f}%")
                else:
                    print(f"  No activation samples for {name}, using dummy data")
                    dummy_input = torch.randn(1, module.in_features).to(device) * 0.1
                    module.a_quantizer.calibrate(dummy_input)
                
                # Mark as calibrated
                module.calibrated = True
                
                # Test weight quantization
                w_quant = module.w_quantizer(module.weight)
                w_sparsity = (w_quant == 0).float().mean().item()
                print(f"  Weight quantizer sparsity: {w_sparsity*100:.2f}%")
                
                calibrated_count += 1
            except Exception as e:
                print(f"Error calibrating {name}: {str(e)}")
    
    print(f"Successfully calibrated {calibrated_count} layers")
    
    # Third pass: Set all layers to quantization mode
    for name, module in model.named_modules():
        if hasattr(module, "mode") and hasattr(module, "calibrated"):
            if module.calibrated:
                # Set mode to use quantization
                module.mode = "quant_forward"
                print(f"Enabled quantization for {name}")
            else:
                print(f"WARNING: Layer {name} not properly calibrated!")

def measure_inference_speed(model, tokenizer, device="cpu", num_runs=50):
    """Measure inference speed for BERT models"""
    model.eval()
    
    # Prepare input for BERT
    text = "This is a test sentence for measuring inference speed."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Warm-up runs
    for _ in range(10):
        with torch.no_grad():
            _ = model(**inputs)
    
    # Timed runs
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(**inputs)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    
    return avg_time

def apply_adaptive_quantization(model, base_sparsity=0.3):
    """Apply layer-specific quantization settings with better scale preservation"""
    layer_configs = {}
    
    # First identify the most problematic layers
    problematic_layers = [
        "cls.predictions.decoder",
        "cls.predictions.transform.dense",
        "bert.encoder.layer.11.output.dense",
        "bert.encoder.layer.10.output.dense",
        "bert.encoder.layer.9.output.dense",
        "bert.encoder.layer.8.output.dense",
        "bert.encoder.layer.7.output.dense",
        "bert.encoder.layer.6.output.dense",
        "bert.encoder.layer.5.output.dense",
        "bert.encoder.layer.4.intermediate.dense",
        "bert.encoder.layer.5.intermediate.dense",
        "bert.encoder.layer.6.intermediate.dense"
    ]
    
    for name, module in model.named_modules():
        if hasattr(module, "mode") and hasattr(module, "calibrated"):
            # Set default quantization parameters
            config = {
                "mode": "weight_only",  # Default mode
                "w_bits": 8,            # Default weight bits
                "sparsity": base_sparsity
            }
            
            # Handle most problematic layers - set to raw mode
            if any(problem_name in name for problem_name in problematic_layers[:3]):
                config["mode"] = "raw"
                print(f"Setting critical layer to raw mode: {name}")
            
            # For intermediate layers in deeper parts of the network
            elif "intermediate.dense" in name:
                # Extract layer number correctly
                if ".layer." in name:
                    parts = name.split(".")
                    for i, part in enumerate(parts):
                        if part == "layer" and i+1 < len(parts):
                            try:
                                layer_num = int(parts[i+1])
                                if layer_num > 3:  # Later layers are more sensitive
                                    # Try raw mode for these as well
                                    config["mode"] = "raw"
                                    print(f"Setting intermediate layer to raw mode: {name}")
                                break
                            except ValueError:
                                # Handle the case where conversion fails
                                print(f"Warning: Could not extract layer number from {name}")
            
            # Apply configuration
            layer_configs[name] = config
    
    # Apply all configurations
    for name, module in model.named_modules():
        if hasattr(module, "mode") and hasattr(module, "calibrated") and name in layer_configs:
            config = layer_configs[name]
            
            # Set mode
            module.mode = config["mode"]
            print(f"Set {name} to mode: {config['mode']}")
            
            # Safety - don't modify bits for raw mode
            if config["mode"] == "raw":
                continue
                
            # Update bits if different from current
            if hasattr(module, "w_quantizer") and config["w_bits"] != module.w_quantizer.n_bits:
                print(f"  Skipping bit-width change for layer: {name} - potentially unstable")
                
    return model


# def apply_adaptive_quantization(model, base_sparsity=0.3):
#     """Apply layer-specific quantization settings"""
#     layer_configs = {}
    
#     for name, module in model.named_modules():
#         if hasattr(module, "mode") and hasattr(module, "calibrated"):
#             # Set default quantization parameters
#             config = {
#                 "mode": "weight_only",  # Default mode
#                 "w_bits": 8,            # Default weight bits
#                 "sparsity": base_sparsity
#             }
            
#             # Customize based on layer type/position
#             if "predictions.decoder" in name:
#                 # Skip quantization for the most problematic layer
#                 config["mode"] = "raw"
#                 print(f"Skipping quantization for critical layer: {name}")
            
#             elif "predictions.transform" in name:
#                 # Use more bits and lower sparsity for embeddings
#                 config["w_bits"] = 16
#                 config["sparsity"] = base_sparsity * 0.5
#                 print(f"Using higher precision for layer: {name}")
                
#             elif "intermediate.dense" in name:
#                 # These layers showed high error - use higher precision
#                 # Extract layer number correctly
#                 if ".layer." in name:
#                     parts = name.split(".")
#                     for i, part in enumerate(parts):
#                         if part == "layer" and i+1 < len(parts):
#                             try:
#                                 layer_num = int(parts[i+1])
#                                 if layer_num > 3:  # Later layers are more sensitive
#                                     config["w_bits"] = 16
#                                     config["sparsity"] = base_sparsity * 0.7
#                                     print(f"Using higher precision for intermediate layer: {name}")
#                                 break
#                             except ValueError:
#                                 # Handle the case where conversion fails
#                                 print(f"Warning: Could not extract layer number from {name}")
            
#             # Apply configuration
#             layer_configs[name] = config
    
#     # Apply all configurations
#     for name, module in model.named_modules():
#         if hasattr(module, "mode") and hasattr(module, "calibrated") and name in layer_configs:
#             config = layer_configs[name]
            
#             # Set mode
#             module.mode = config["mode"]
#             print(f"Set {name} to mode: {config['mode']}")
            
#             # Update bits if different from current
#             if hasattr(module, "w_quantizer") and config["w_bits"] != module.w_quantizer.n_bits:
#                 original_scale = module.w_quantizer.scale.clone()
                
#                 # Create new quantizer with updated bits
#                 from tubgemm_quantizers.tublog_quantizer import TubLogQuantizer
#                 module.w_quantizer = TubLogQuantizer(
#                     n_bits=config["w_bits"],
#                     symmetric=False,
#                     channel_wise=module.w_quantizer.channel_wise,
#                     sparsity_threshold=config["sparsity"]
#                 )
                
#                 # Recalibrate with original weight
#                 module.w_quantizer.scale.data = original_scale
#                 module.w_quantizer.calibrate(module.weight)
#                 print(f"  Updated w_bits to {config['w_bits']} and recalibrated")
                
#     return model

def test_layer_by_layer(model, tokenizer, device="cpu"):
    """Test each layer individually to find problematic ones"""
    print("\n=== Layer-by-Layer Quantization Test ===")
    
    # Start with all layers in raw mode
    for name, module in model.named_modules():
        if hasattr(module, "mode"):
            module.mode = "raw"
    
    # Prepare test input
    text = "This is a test sentence for measuring inference."
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Get baseline output with no quantization
    with torch.no_grad():
        baseline_output = model(**inputs).logits
    
    # Test each layer individually
    for i, (name, module) in enumerate(model.named_modules()):
        if hasattr(module, "mode") and hasattr(module, "calibrated"):
            # Reset all layers to raw
            for reset_name, reset_module in model.named_modules():
                if hasattr(reset_module, "mode"):
                    reset_module.mode = "raw"
            
            # Set only this layer to quantized
            module.mode = "weight_only"  # Test with weight-only quantization
            
            # Run inference
            with torch.no_grad():
                try:
                    layer_output = model(**inputs).logits
                    
                    # Calculate difference from baseline
                    diff = (baseline_output - layer_output).abs().mean().item()
                    
                    # Check for NaNs or extreme values
                    has_nan = torch.isnan(layer_output).any()
                    has_inf = torch.isinf(layer_output).any()
                    
                    print(f"Layer {i}: {name}")
                    print(f"  Output diff: {diff:.6f}")
                    print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
                    
                except Exception as e:
                    print(f"Layer {i}: {name} - ERROR: {str(e)}")
            
            # Reset back to raw mode
            module.mode = "raw"
    
    # Test for accumulated effect
    print("\n=== Testing Progressive Quantization ===")
    quantized_count = 0
    
    # Categorize layers by depth
    layers_by_depth = []
    for name, module in model.named_modules():
        if hasattr(module, "mode") and hasattr(module, "calibrated"):
            layers_by_depth.append((name, module))
    
    # Start with all raw
    for name, module in model.named_modules():
        if hasattr(module, "mode"):
            module.mode = "raw"
    
    # Progressively enable quantization in more layers
    step_size = max(1, len(layers_by_depth) // 5)  # Test in ~5 steps
    for step in range(0, len(layers_by_depth), step_size):
        # Enable quantization for layers up to this point
        for i in range(step + 1):
            if i < len(layers_by_depth):
                name, module = layers_by_depth[i]
                module.mode = "weight_only"
                quantized_count += 1
        
        # Evaluate
        with torch.no_grad():
            try:
                step_output = model(**inputs).logits
                diff = (baseline_output - step_output).abs().mean().item()
                has_nan = torch.isnan(step_output).any()
                has_inf = torch.isinf(step_output).any()
                
                print(f"Step {step//step_size + 1}: Quantized {quantized_count}/{len(layers_by_depth)} layers")
                print(f"  Output diff: {diff:.6f}")
                print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
                
            except Exception as e:
                print(f"Step {step//step_size + 1}: ERROR - {str(e)}")
        
        # Reset quantized count for next step
        quantized_count = 0

def main():
    # Define device
    device = "cpu"
    print(f"Using device: {device}")
    
    # Test texts for perplexity evaluation
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models have revolutionized natural language processing.",
        "Quantization helps reduce model size while maintaining performance.",
        "Neural networks can learn complex patterns from data.",
        "The transformer architecture has enabled significant progress in NLP.",
        "Computer vision systems can now recognize objects with high accuracy.",
        "Reinforcement learning algorithms learn through trial and error.",
        "BERT models are based on the transformer architecture.",
        "Deep learning has applications in many different fields.",
        "Model compression techniques help deploy AI on resource-constrained devices."
    ]
    
    # Calibration texts
    calib_texts = [
        "The transformer architecture has revolutionized NLP.",
        "Quantization techniques can significantly reduce model size.",
        "Transfer learning enables models to leverage pre-existing knowledge.",
        "Attention mechanisms help models focus on relevant parts of the input.",
        "Language models have achieved state-of-the-art results on many tasks.",
        "Model distillation creates smaller models that retain performance.",
        "Fine-tuning adapts pre-trained models to specific downstream tasks.",
        "Language models can generate coherent and contextually relevant text.",
        "Self-supervised learning reduces the need for labeled training data.",
        "Transformer models process all tokens in parallel during training.",
        "BERT is a bidirectional transformer model for various NLP tasks.",
        "Tokens in the input sequence attend to all other tokens."
    ]
    
    # Load BERT model
    model_name = "bert-base-uncased"  # Choose any BERT model you prefer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    # Print original model size
    print(f"Original model size: {model.get_memory_footprint():,} bytes")
    
    # Move model to device
    model = model.to(device)
    
    # Calculate original perplexity
    print("\nCalculating original perplexity...")
    try:
        orig_perplexity = calculate_bert_perplexity(model, tokenizer, test_texts, device)
        print(f"Original perplexity: {orig_perplexity:.2f}")
    except Exception as e:
        print(f"Error calculating original perplexity: {str(e)}")
        orig_perplexity = float("inf")
    
    # Measure original inference speed
    print("\nMeasuring original inference speed...")
    orig_time = measure_inference_speed(model, tokenizer, device)
    print(f"Original inference time: {orig_time*1000:.2f} ms per sample")
    
    # ===== 2. Quantize the model =====
    print("\nQuantizing model...")
    # Increase sparsity threshold to create more zeros
    sparsity_threshold = 0.3  # Higher value to induce more sparsity
    
    # Convert the model to use TubLog quantization layers
    quant_model = convert_model_to_tublog(model, w_bit=8, a_bit=8, sparsity_threshold=sparsity_threshold)
    
    # Print quantized model size
    print(f"Quantized model size: {quant_model.get_memory_footprint():,} bytes")
    print(f"Size reduction: {(1 - quant_model.get_memory_footprint() / model.get_memory_footprint()) * 100:.2f}%")
    
    # ===== 3. Calibrate the quantized model =====
    print("\nCalibrating quantized model...")
    # Create calibration dataset
    calib_dataset = BertCalibrationDataset(tokenizer, calib_texts)
    calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=2)
    
    # Calibrate the model
    calibrate_bert_model(quant_model, calib_loader, device)

    # After calibration
    print("\nSwitching to weight-only quantization for better results...")
    for name, module in quant_model.named_modules():
        if hasattr(module, "mode") and hasattr(module, "calibrated"):
            if module.calibrated:
                module.mode = "weight_only"
                print(f"Enabled weight-only quantization for {name}")

    # Run layer-by-layer testing to identify problematic layers
    test_layer_by_layer(quant_model, tokenizer, device)

    # After identifying problematic layers, you can try to fix them or exclude them
    print("\nCalculating perplexity with weight-only quantization...")
    weight_only_perplexity = calculate_bert_perplexity(quant_model, tokenizer, test_texts, device)
    print(f"Weight-only perplexity: {weight_only_perplexity:.2f}")
    if orig_perplexity != float("inf"):
        ppl_diff = (weight_only_perplexity / orig_perplexity - 1) * 100
        print(f"Perplexity increase: {ppl_diff:.2f}%")

    # Apply adaptive quantization based on layer sensitivity
    print("\nApplying adaptive quantization strategy...")
    quant_model = apply_adaptive_quantization(quant_model, base_sparsity=0.2)

    # Test with adaptive quantization
    print("\nCalculating perplexity with adaptive quantization...")
    adaptive_perplexity = calculate_bert_perplexity(quant_model, tokenizer, test_texts, device)
    print(f"Adaptive quantization perplexity: {adaptive_perplexity:.2f}")
    if orig_perplexity != float("inf"):
        ppl_diff = (adaptive_perplexity / orig_perplexity - 1) * 100
        print(f"Perplexity increase: {ppl_diff:.2f}%")
    
    print("\n=== Layer Mode Summary ===")
    mode_counts = {}
    for name, module in quant_model.named_modules():
        if hasattr(module, "mode"):
            mode = module.mode
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            if "predictions" in name or ".output.dense" in name or ".intermediate.dense" in name:
                print(f"Layer {name}: mode={mode}")

    print(f"\nMode distribution: {mode_counts}")
        
    # ===== 4. Post-calibration stats =====
    print("\n=== After calibration ===")
    print_tublog_statistics(quant_model)
    
    # ===== 5. Calculate quantized perplexity =====
    print("\nCalculating final quantized perplexity...")
    try:
        # Don't override the modes that were set by adaptive quantization
        # Just make sure all layers are calibrated
        for name, module in quant_model.named_modules():
            if hasattr(module, "mode") and hasattr(module, "calibrated"):
                if not module.calibrated:
                    print(f"Warning: Layer {name} not calibrated before final evaluation!")
                    
        quant_perplexity = calculate_bert_perplexity(quant_model, tokenizer, test_texts, device)
        print(f"Final quantized perplexity: {quant_perplexity:.2f}")
        
        # Compare with original
        if orig_perplexity != float("inf"):
            ppl_diff = (quant_perplexity / orig_perplexity - 1) * 100
            print(f"Final perplexity increase: {ppl_diff:.2f}%")
    except Exception as e:
        print(f"Error calculating quantized perplexity: {str(e)}")
    
    # Measure quantized inference speed
    print("\nMeasuring quantized inference speed...")
    quant_time = measure_inference_speed(quant_model, tokenizer, device)
    print(f"Quantized inference time: {quant_time*1000:.2f} ms per sample")
    print(f"Speedup: {orig_time/quant_time:.2f}x")
    
    # ===== 6. Debug individual layers =====
    print("\n=== Layer-by-layer Analysis ===")
    for name, module in quant_model.named_modules():
        if hasattr(module, "quant_forward") and hasattr(module, "calibrated"):
            if module.calibrated:
                # Generate sample input
                if hasattr(module, "in_features"):
                    # Standard case - create random input
                    sample_input = torch.randn(1, module.in_features).to(device)
                    
                    try:
                        # Compare different modes
                        module.mode = "raw"
                        raw_out = module(sample_input)
                        
                        module.mode = "quant_forward"
                        quant_out = module(sample_input)
                        
                        diff = (raw_out - quant_out).abs().mean().item()
                        sparsity = (quant_out == 0).float().mean().item()
                        
                        print(f"Layer {name}:")
                        print(f"  Output diff: {diff:.6f}")
                        print(f"  Output sparsity: {sparsity*100:.2f}%")
                        
                        # Check for all-zero outputs
                        if torch.all(quant_out == 0):
                            print(f"  WARNING: All outputs are zero!")
                            
                    except Exception as e:
                        print(f"  Error in layer analysis: {str(e)}")

if __name__ == "__main__":
    main()