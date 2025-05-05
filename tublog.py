# main.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tubgemm_utils.model_converter import convert_model_to_tublog, calibrate_tublog_model, print_tublog_statistics
from tqdm.auto import tqdm
from tubgemm_utils.model_converter import print_q_parameter_values

class TubGEMMCalibrationDataset(torch.utils.data.Dataset):
    """Simple dataset for calibration"""
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
        
        # Return just the input_ids tensor without batch dimension
        return encoding["input_ids"].squeeze(0)

def calculate_perplexity(model, tokenizer, test_texts, device="cpu"):
    """Calculate pseudo-perplexity for sequence classification models"""
    model.eval()
    model.to(device)
    
    perplexities = []
    
    for text in tqdm(test_texts, desc="Calculating perplexity"):
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                # For sequence classification, use a pseudo-perplexity
                probs = torch.nn.functional.softmax(logits, dim=-1)
                max_probs = probs.max(dim=-1).values
                perplexity = (1 / max_probs.mean()).item()
                perplexities.append(perplexity)
                
        except Exception as e:
            print(f"Error processing text: {text[:50]}... - {str(e)}")
            perplexities.append(float('inf'))
    
    return sum(perplexities) / len(perplexities) if perplexities else float('inf')

def main():
    # Define device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    device = torch.device("cpu")
    print(f"Using device: {device}")
    # Test texts for perplexity evaluation
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models have revolutionized natural language processing.",
        "Quantization helps reduce model size while maintaining performance.",
        "TinyBERT is a distilled version of BERT that is more efficient."
    ]
    
    # Calibration texts
    calib_texts = [
        "The transformer architecture has revolutionized NLP.",
        "Quantization techniques can significantly reduce model size.",
        "Transfer learning enables models to leverage pre-existing knowledge.",
        "Attention mechanisms help models focus on relevant parts of the input.",
        "BERT models have achieved state-of-the-art results on many NLP tasks.",
        "Model distillation creates smaller models that retain performance.",
        "Fine-tuning adapts pre-trained models to specific downstream tasks.",
        "Language models can generate coherent and contextually relevant text."
    ]
    
    # Load model
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # ===== 1. Pre-quantization weight check =====
    print("\n=== Before quantization ===")
    query_layer = model.bert.encoder.layer[0].attention.self.query
    print(f"\nbert.encoder.layer.0.attention.self.query weights:")
    print("Original (first 3x3):\n", query_layer.weight.data[:3, :3])
    
    # Calculate original perplexity
    print("\nCalculating original perplexity...")
    orig_perplexity = calculate_perplexity(model, tokenizer, test_texts, device)
    print(f"Original perplexity: {orig_perplexity:.2f}")
    
    # ===== 2. Quantize the model =====
    print("\nQuantizing model...")
    # Convert the model to use TubLog quantization layers
    quant_model = convert_model_to_tublog(model, w_bit=4, a_bit=4, sparsity_threshold=0.01)
    #fix later
    
    # Check weights after conversion
    print("\n=== After quantization ===")
    query_layer = quant_model.bert.encoder.layer[0].attention.self.query
    print(f"\nbert.encoder.layer.0.attention.self.query weights:")
    print("Original (first 3x3):\n", query_layer.weight.data[:3, :3])
    
    # ===== 3. Calibrate the quantized model =====
    print("\nCalibrating quantized model...")
    # Create calibration dataset
    calib_dataset = TubGEMMCalibrationDataset(tokenizer, calib_texts)
    calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=2)
    
    # Calibrate the model
    calibrate_tublog_model(quant_model, calib_loader, device, is_encoder_decoder=False)

    
    # ===== 4. Post-calibration weight check =====
    print("\n=== After calibration ===")
    print_tublog_statistics(quant_model)
    print_q_parameter_values(quant_model)
    
    # ===== 5. Calculate quantized perplexity =====
    print("\nCalculating quantized perplexity...")
    quant_perplexity = calculate_perplexity(quant_model, tokenizer, test_texts, device)
    print(f"Quantized perplexity: {quant_perplexity:.2f}")
    
    # Save the quantized model
    torch.save(quant_model.state_dict(), "tublog_tinybert.pt")
    print("\nQuantized model saved to tublog_tinybert.pt")

if __name__ == "__main__":
    main()