"""
Modern Paraphrase Generation System

This module provides state-of-the-art paraphrase generation capabilities using
Hugging Face transformers, with support for multiple models and techniques.
"""

import logging
import random
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    pipeline,
    T5ForConditionalGeneration,
    T5Tokenizer
)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import evaluate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ParaphraseConfig:
    """Configuration for paraphrase generation."""
    model_name: str = "tuner007/pegasus_paraphrase"
    max_length: int = 100
    num_return_sequences: int = 3
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    device: str = "auto"


class ParaphraseGenerator:
    """
    Modern paraphrase generation system with multiple model support.
    
    Supports various state-of-the-art models for paraphrase generation including
    Pegasus, T5, and BART models with advanced sampling techniques.
    """
    
    def __init__(self, config: Optional[ParaphraseConfig] = None):
        """
        Initialize the paraphrase generator.
        
        Args:
            config: Configuration object for paraphrase generation
        """
        self.config = config or ParaphraseConfig()
        self.model = None
        self.tokenizer = None
        self.device = self._get_device()
        self._load_model()
        
        # Initialize evaluation metrics
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")
        
    def _get_device(self) -> str:
        """Determine the best available device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.config.device
    
    def _load_model(self) -> None:
        """Load the specified model and tokenizer."""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Try different model loading strategies
            if "t5" in self.config.model_name.lower():
                self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.warning(f"Failed to load {self.config.model_name}: {e}")
            logger.info("Falling back to pipeline approach")
            self._load_pipeline_model()
    
    def _load_pipeline_model(self) -> None:
        """Fallback to pipeline-based model loading."""
        try:
            self.pipeline_model = pipeline(
                "text2text-generation",
                model=self.config.model_name,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Pipeline model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load pipeline model: {e}")
            raise RuntimeError("Could not load any paraphrase model")
    
    def generate_paraphrases(
        self, 
        text: str, 
        num_paraphrases: Optional[int] = None,
        prefix: str = "paraphrase: "
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Generate paraphrases for the given text.
        
        Args:
            text: Input text to paraphrase
            num_paraphrases: Number of paraphrases to generate
            prefix: Prefix to add to input text (model-specific)
            
        Returns:
            List of dictionaries containing paraphrases and their scores
        """
        num_paraphrases = num_paraphrases or self.config.num_return_sequences
        
        try:
            if hasattr(self, 'pipeline_model'):
                return self._generate_with_pipeline(text, num_paraphrases, prefix)
            else:
                return self._generate_with_model(text, num_paraphrases, prefix)
        except Exception as e:
            logger.error(f"Error generating paraphrases: {e}")
            return []
    
    def _generate_with_pipeline(
        self, 
        text: str, 
        num_paraphrases: int, 
        prefix: str
    ) -> List[Dict[str, Union[str, float]]]:
        """Generate paraphrases using Hugging Face pipeline."""
        input_text = f"{prefix}{text}"
        
        results = self.pipeline_model(
            input_text,
            max_length=self.config.max_length,
            num_return_sequences=num_paraphrases,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            pad_token_id=self.pipeline_model.tokenizer.eos_token_id
        )
        
        paraphrases = []
        for i, result in enumerate(results):
            paraphrase = result['generated_text'].strip()
            # Calculate similarity score
            similarity = self._calculate_similarity(text, paraphrase)
            paraphrases.append({
                'text': paraphrase,
                'similarity_score': similarity,
                'rank': i + 1
            })
        
        return paraphrases
    
    def _generate_with_model(
        self, 
        text: str, 
        num_paraphrases: int, 
        prefix: str
    ) -> List[Dict[str, Union[str, float]]]:
        """Generate paraphrases using direct model inference."""
        input_text = f"{prefix}{text}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate paraphrases
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.max_length,
                num_return_sequences=num_paraphrases,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        paraphrases = []
        for i, output in enumerate(outputs):
            paraphrase = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove prefix if present
            if paraphrase.startswith(prefix):
                paraphrase = paraphrase[len(prefix):].strip()
            
            similarity = self._calculate_similarity(text, paraphrase)
            paraphrases.append({
                'text': paraphrase,
                'similarity_score': similarity,
                'rank': i + 1
            })
        
        return paraphrases
    
    def _calculate_similarity(self, original: str, paraphrase: str) -> float:
        """
        Calculate semantic similarity between original and paraphrase.
        
        Uses a simple word overlap approach. In production, you might want to
        use sentence embeddings for better semantic similarity.
        """
        original_words = set(original.lower().split())
        paraphrase_words = set(paraphrase.lower().split())
        
        if not original_words or not paraphrase_words:
            return 0.0
        
        intersection = original_words.intersection(paraphrase_words)
        union = original_words.union(paraphrase_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def evaluate_paraphrases(
        self, 
        original: str, 
        paraphrases: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate paraphrases using ROUGE and BLEU metrics.
        
        Args:
            original: Original text
            paraphrases: List of generated paraphrases
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not paraphrases:
            return {}
        
        # Calculate ROUGE scores
        rouge_scores = self.rouge.compute(
            predictions=paraphrases,
            references=[original] * len(paraphrases)
        )
        
        # Calculate BLEU scores
        bleu_scores = []
        for paraphrase in paraphrases:
            bleu_score = self.bleu.compute(
                predictions=[paraphrase.split()],
                references=[[original.split()]]
            )
            bleu_scores.append(bleu_score['bleu'])
        
        return {
            'rouge_1': rouge_scores['rouge1'],
            'rouge_2': rouge_scores['rouge2'],
            'rouge_l': rouge_scores['rougel'],
            'bleu_mean': np.mean(bleu_scores),
            'bleu_std': np.std(bleu_scores)
        }
    
    def batch_paraphrase(
        self, 
        texts: List[str], 
        num_paraphrases: int = 1
    ) -> List[List[Dict[str, Union[str, float]]]]:
        """
        Generate paraphrases for a batch of texts.
        
        Args:
            texts: List of input texts
            num_paraphrases: Number of paraphrases per text
            
        Returns:
            List of paraphrase results for each input text
        """
        results = []
        for text in texts:
            paraphrases = self.generate_paraphrases(text, num_paraphrases)
            results.append(paraphrases)
        return results


def create_synthetic_dataset(size: int = 100) -> List[str]:
    """
    Create a synthetic dataset for testing paraphrase generation.
    
    Args:
        size: Number of sentences to generate
        
    Returns:
        List of synthetic sentences
    """
    templates = [
        "The {adjective} {noun} {verb} over the {adjective2} {noun2}.",
        "In the {place}, {person} {verb} {object}.",
        "{Person} is {adjective} because {reason}.",
        "The {weather} weather makes {person} feel {emotion}.",
        "{Person} {verb} {object} in the {place}.",
        "When {event} happens, {person} {reaction}.",
        "The {animal} {verb} through the {location}.",
        "{Person} {verb} {object} with {tool}.",
        "In {time}, {person} will {future_action}.",
        "The {color} {object} {verb} {direction}."
    ]
    
    vocabulary = {
        'adjective': ['quick', 'slow', 'bright', 'dark', 'large', 'small', 'happy', 'sad'],
        'noun': ['fox', 'dog', 'cat', 'bird', 'car', 'house', 'tree', 'flower'],
        'verb': ['jumps', 'runs', 'flies', 'walks', 'drives', 'grows', 'blooms', 'falls'],
        'adjective2': ['lazy', 'active', 'sleepy', 'energetic', 'quiet', 'loud'],
        'noun2': ['dog', 'cat', 'mouse', 'rabbit', 'squirrel', 'bird'],
        'place': ['park', 'garden', 'kitchen', 'office', 'school', 'library'],
        'person': ['John', 'Mary', 'Alex', 'Sarah', 'Mike', 'Emma'],
        'object': ['book', 'ball', 'phone', 'computer', 'cup', 'pen'],
        'reason': ['it is sunny', 'they are tired', 'work is done', 'friends are coming'],
        'weather': ['sunny', 'rainy', 'cloudy', 'windy', 'snowy'],
        'emotion': ['happy', 'sad', 'excited', 'calm', 'worried'],
        'event': ['it rains', 'the sun shines', 'school ends', 'dinner is ready'],
        'reaction': ['smiles', 'dances', 'sings', 'celebrates', 'relaxes'],
        'animal': ['rabbit', 'squirrel', 'deer', 'bird', 'butterfly'],
        'location': ['forest', 'meadow', 'garden', 'park', 'field'],
        'tool': ['hammer', 'brush', 'pen', 'camera', 'phone'],
        'time': ['tomorrow', 'next week', 'next month', 'next year'],
        'future_action': ['travel', 'graduate', 'move', 'start a new job'],
        'color': ['red', 'blue', 'green', 'yellow', 'purple', 'orange'],
        'direction': ['up', 'down', 'left', 'right', 'forward', 'backward']
    }
    
    sentences = []
    for _ in range(size):
        template = random.choice(templates)
        sentence = template
        
        # Replace placeholders with random vocabulary
        for key, values in vocabulary.items():
            placeholder = f"{{{key}}}"
            if placeholder in sentence:
                sentence = sentence.replace(placeholder, random.choice(values), 1)
        
        sentences.append(sentence)
    
    return sentences


if __name__ == "__main__":
    # Example usage
    config = ParaphraseConfig(
        model_name="tuner007/pegasus_paraphrase",
        num_return_sequences=3,
        temperature=0.8
    )
    
    generator = ParaphraseGenerator(config)
    
    # Test with sample sentences
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the way we work.",
        "The weather is beautiful today."
    ]
    
    for sentence in test_sentences:
        print(f"\nOriginal: {sentence}")
        paraphrases = generator.generate_paraphrases(sentence)
        
        for i, result in enumerate(paraphrases, 1):
            print(f"Paraphrase {i}: {result['text']}")
            print(f"Similarity: {result['similarity_score']:.3f}")
