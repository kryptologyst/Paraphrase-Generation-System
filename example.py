#!/usr/bin/env python3
"""
Example script demonstrating the paraphrase generation system.

This script shows how to use the system programmatically with different
models and configurations.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from paraphrase_generator import ParaphraseGenerator, ParaphraseConfig, create_synthetic_dataset


def demo_basic_usage():
    """Demonstrate basic usage of the paraphrase generator."""
    print("=" * 60)
    print("BASIC USAGE DEMO")
    print("=" * 60)
    
    # Create configuration
    config = ParaphraseConfig(
        model_name="tuner007/pegasus_paraphrase",
        num_return_sequences=3,
        temperature=0.7
    )
    
    # Initialize generator
    print("Initializing paraphrase generator...")
    generator = ParaphraseGenerator(config)
    
    # Test sentences
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the way we work.",
        "The weather is beautiful today.",
        "Python is a versatile programming language.",
        "Artificial intelligence will change the future."
    ]
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n{i}. Original: {sentence}")
        
        try:
            paraphrases = generator.generate_paraphrases(sentence)
            
            for j, result in enumerate(paraphrases, 1):
                print(f"   Paraphrase {j}: {result['text']}")
                print(f"   Similarity: {result['similarity_score']:.3f}")
        
        except Exception as e:
            print(f"   Error: {e}")


def demo_different_models():
    """Demonstrate different model configurations."""
    print("\n" + "=" * 60)
    print("DIFFERENT MODELS DEMO")
    print("=" * 60)
    
    models = [
        ("tuner007/pegasus_paraphrase", "Pegasus (Recommended)"),
        ("t5-small", "T5 Small (Fast)"),
        ("t5-base", "T5 Base (Balanced)")
    ]
    
    test_text = "The quick brown fox jumps over the lazy dog."
    
    for model_name, description in models:
        print(f"\n{description} ({model_name}):")
        
        try:
            config = ParaphraseConfig(
                model_name=model_name,
                num_return_sequences=2,
                temperature=0.8
            )
            
            generator = ParaphraseGenerator(config)
            paraphrases = generator.generate_paraphrases(test_text)
            
            for i, result in enumerate(paraphrases, 1):
                print(f"  {i}. {result['text']}")
        
        except Exception as e:
            print(f"  Error loading {model_name}: {e}")


def demo_synthetic_data():
    """Demonstrate synthetic data generation."""
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA DEMO")
    print("=" * 60)
    
    print("Generating 5 synthetic sentences...")
    sentences = create_synthetic_dataset(5)
    
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")


def demo_evaluation_metrics():
    """Demonstrate evaluation metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION METRICS DEMO")
    print("=" * 60)
    
    config = ParaphraseConfig(
        model_name="tuner007/pegasus_paraphrase",
        num_return_sequences=3
    )
    
    generator = ParaphraseGenerator(config)
    
    original = "The quick brown fox jumps over the lazy dog."
    paraphrases = generator.generate_paraphrases(original)
    
    if paraphrases:
        paraphrase_texts = [p['text'] for p in paraphrases]
        metrics = generator.evaluate_paraphrases(original, paraphrase_texts)
        
        print(f"Original: {original}")
        print(f"\nParaphrases:")
        for i, text in enumerate(paraphrase_texts, 1):
            print(f"  {i}. {text}")
        
        print(f"\nEvaluation Metrics:")
        print(f"  ROUGE-1: {metrics['rouge_1']:.3f}")
        print(f"  ROUGE-2: {metrics['rouge_2']:.3f}")
        print(f"  ROUGE-L: {metrics['rouge_l']:.3f}")
        print(f"  BLEU (Mean): {metrics['bleu_mean']:.3f}")
        print(f"  BLEU (Std): {metrics['bleu_std']:.3f}")


def demo_batch_processing():
    """Demonstrate batch processing."""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING DEMO")
    print("=" * 60)
    
    config = ParaphraseConfig(
        model_name="tuner007/pegasus_paraphrase",
        num_return_sequences=2
    )
    
    generator = ParaphraseGenerator(config)
    
    texts = [
        "Hello world!",
        "Machine learning is amazing.",
        "Python is great for data science."
    ]
    
    print("Processing batch of texts...")
    results = generator.batch_paraphrase(texts, num_paraphrases=2)
    
    for i, (text, paraphrases) in enumerate(zip(texts, results), 1):
        print(f"\n{i}. Original: {text}")
        for j, result in enumerate(paraphrases, 1):
            print(f"   Paraphrase {j}: {result['text']}")


def main():
    """Run all demonstrations."""
    print("Paraphrase Generation System - Demo")
    print("This demo showcases the capabilities of the system.")
    print("Note: First run may take time to download models.")
    
    try:
        demo_basic_usage()
        demo_different_models()
        demo_synthetic_data()
        demo_evaluation_metrics()
        demo_batch_processing()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Try the web interface: streamlit run web_app/app.py")
        print("2. Use the CLI: python src/cli.py 'Your text here'")
        print("3. Check the README.md for more examples")
        
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        print("Please check your installation and try again.")


if __name__ == "__main__":
    main()
