import torch
import torch.nn.functional as F
from tqdm import tqdm

def calculate_perplexity(model, tokenizer, test_texts, device="cpu"):  # Changed to CPU by default
    """
    Calculate pseudo-perplexity for sequence classification models
    """
    model.eval()
    model.to(device)
    
    perplexities = []
    
    for text in tqdm(test_texts, desc="Calculating perplexity"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            try:
                outputs = model(**inputs)
                logits = outputs.logits
                
                # For sequence classification, we'll use a pseudo-perplexity
                # by looking at the confidence of the predicted class
                probs = F.softmax(logits, dim=-1)
                max_probs = probs.max(dim=-1).values
                perplexity = (1 / max_probs.mean()).item()
                perplexities.append(perplexity)
                
            except Exception as e:
                print(f"Error processing text: {text[:50]}... - {str(e)}")
                perplexities.append(float('inf'))  # Use infinity for failed cases
    
    return sum(perplexities) / len(perplexities) if perplexities else float('inf')