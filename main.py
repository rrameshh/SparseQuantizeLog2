from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llm_utils.datasets import LLMCalibrationLoader
from quant_tinybert import quantize_tinybert
from llm_utils.calibrator import TinyBertQuantCalibrator
from perplexity import calculate_perplexity
import torch

def get_device():
    # return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"

def main():
    device = get_device()
    print(f"Using device: {device}")
    
    # Test texts for perplexity evaluation
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models have revolutionized natural language processing.",
        "Quantization helps reduce model size while maintaining performance.",
        "TinyBERT is a distilled version of BERT that is more efficient."
    ]
    
    # Load model
    model_name = "huawei-noah/TinyBERT_General_4L_312D"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Calculate original perplexity
    print("\nCalculating original perplexity...")
    orig_perplexity = calculate_perplexity(model, tokenizer, test_texts, device)
    print(f"Original perplexity: {orig_perplexity:.2f}")
    
    # Quantize
    print("\nQuantizing model...")
    quant_model = quantize_tinybert(model)
    
    # Calibrate
    print("Calibrating quantized model...")
    calib_loader = LLMCalibrationLoader(tokenizer)
    calib_dataloader = calib_loader.get_calibration_loader(num_samples=32)
    calibrator = TinyBertQuantCalibrator(quant_model, calib_dataloader)
    calibrator.llm_quant_calib()
    
    # Calculate quantized perplexity
    print("\nCalculating quantized perplexity...")
    quant_perplexity = calculate_perplexity(quant_model, tokenizer, test_texts, device)
    print(f"Quantized perplexity: {quant_perplexity:.2f}")
    
    # Save results
    # results = {
    #     "original_perplexity": orig_perplexity,
    #     "quantized_perplexity": quant_perplexity,
    #     "test_texts": test_texts,
    #     "device": device
    # }
    
    # torch.save(results, "perplexity_results.pt")
    # torch.save(quant_model.state_dict(), "quantized_tinybert.pt")
    
    # print("\nResults saved to perplexity_results.pt")

if __name__ == "__main__":
    main()