# llm_utils/datasets.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class LLMCalibrationDataset(Dataset):
    """
    Dataset for LLM calibration using text sequences.
    """
    def __init__(self, tokenizer, texts, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize input
        tokens = self.tokenizer(
            text, 
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        
        return input_ids, attention_mask

class LLMCalibrationLoader:
    """
    Utility for creating calibration data loaders for LLMs.
    """
    def __init__(self, tokenizer, calib_texts=None, batch_size=4):
        self.tokenizer = tokenizer
        self.calib_texts = calib_texts
        self.batch_size = batch_size
        
    def get_calibration_loader(self, num_samples=32, max_length=512):
        """Create a calibration data loader with diverse text."""
        if self.calib_texts is None or len(self.calib_texts) < num_samples:
            # Generate synthetic diverse text if not provided
            self.calib_texts = self._generate_diverse_texts(num_samples)
            
        # Take subset if needed
        calib_texts = self.calib_texts[:num_samples]
        
        # Create dataset
        dataset = LLMCalibrationDataset(self.tokenizer, calib_texts, max_length)
        
        # Create dataloader
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )
        
    def _generate_diverse_texts(self, num_samples):
        """
        Generate diverse text samples for calibration.
        This ensures we capture a range of linguistic patterns.
        """
        # Example diverse text patterns (would be better with real data)
        templates = [
            "This is a simple sentence about {topic}.",
            "Have you ever wondered about {topic}? I certainly have.",
            "{topic} is one of the most important aspects of modern society.",
            "In the context of {topic}, we must consider multiple perspectives.",
            "The history of {topic} dates back to ancient times.",
            "Scientists recently discovered new facts about {topic}.",
            "Many people don't understand {topic} properly.",
            "{topic}? What a fascinating subject to explore in depth!"
        ]
        
        topics = [
            "artificial intelligence", "climate change", "quantum physics",
            "democracy", "education", "healthcare", "technology", "economics",
            "philosophy", "music", "art", "literature", "history", "mathematics",
            "biology", "chemistry", "psychology", "sociology", "linguistics"
        ]
        
        texts = []
        np.random.seed(42)  # For reproducibility
        
        for _ in range(num_samples):
            template = np.random.choice(templates)
            topic = np.random.choice(topics)
            texts.append(template.format(topic=topic))
            
        return texts