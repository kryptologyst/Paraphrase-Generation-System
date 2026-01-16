"""
Command-line interface for the paraphrase generation system.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from paraphrase_generator import ParaphraseGenerator, ParaphraseConfig, create_synthetic_dataset
from config import ConfigManager


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Paraphrase Generation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "The quick brown fox jumps over the lazy dog."
  python cli.py --file input.txt --output paraphrases.json
  python cli.py --synthetic 10 --model t5-small
  python cli.py "Hello world" --num-paraphrases 5 --temperature 0.8
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        'text',
        nargs='?',
        help='Text to paraphrase'
    )
    input_group.add_argument(
        '--file',
        type=str,
        help='File containing text to paraphrase'
    )
    input_group.add_argument(
        '--synthetic',
        type=int,
        metavar='N',
        help='Generate N synthetic sentences and paraphrase them'
    )
    
    # Model options
    parser.add_argument(
        '--model',
        type=str,
        default='tuner007/pegasus_paraphrase',
        help='Model name to use for paraphrase generation'
    )
    parser.add_argument(
        '--num-paraphrases',
        type=int,
        default=3,
        help='Number of paraphrases to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.9,
        help='Top-p sampling parameter'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=100,
        help='Maximum length of generated text'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (JSON format)'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'text', 'csv'],
        default='text',
        help='Output format'
    )
    parser.add_argument(
        '--show-metrics',
        action='store_true',
        help='Show evaluation metrics'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def load_text_from_file(file_path: str) -> str:
    """Load text from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        sys.exit(1)


def save_results(results: List[Dict[str, Any]], output_path: str, format_type: str):
    """Save results to file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            if format_type == 'json':
                json.dump(results, f, indent=2, ensure_ascii=False)
            elif format_type == 'csv':
                import pandas as pd
                df = pd.DataFrame(results)
                df.to_csv(f, index=False)
            else:  # text format
                for i, result in enumerate(results, 1):
                    f.write(f"Paraphrase {i}: {result['text']}\n")
                    f.write(f"Similarity: {result['similarity_score']:.3f}\n\n")
        
        print(f"Results saved to {output_path}")
    
    except Exception as e:
        print(f"Error saving results: {e}")


def display_results(results: List[Dict[str, Any]], show_metrics: bool = False, generator=None, original: str = ""):
    """Display results in the terminal."""
    if not results:
        print("No paraphrases generated.")
        return
    
    print(f"\n{'='*60}")
    print(f"ORIGINAL: {original}")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        print(f"\nParaphrase {i}:")
        print(f"  Text: {result['text']}")
        print(f"  Similarity Score: {result['similarity_score']:.3f}")
    
    if show_metrics and generator and original:
        print(f"\n{'='*60}")
        print("EVALUATION METRICS")
        print(f"{'='*60}")
        
        paraphrase_texts = [r['text'] for r in results]
        metrics = generator.evaluate_paraphrases(original, paraphrase_texts)
        
        if metrics:
            print(f"ROUGE-1: {metrics['rouge_1']:.3f}")
            print(f"ROUGE-2: {metrics['rouge_2']:.3f}")
            print(f"ROUGE-L: {metrics['rouge_l']:.3f}")
            print(f"BLEU (Mean): {metrics['bleu_mean']:.3f}")
            print(f"BLEU (Std): {metrics['bleu_std']:.3f}")


def process_synthetic_data(generator: ParaphraseGenerator, num_sentences: int, args):
    """Process synthetic data generation and paraphrasing."""
    print(f"Generating {num_sentences} synthetic sentences...")
    sentences = create_synthetic_dataset(num_sentences)
    
    all_results = []
    
    for i, sentence in enumerate(sentences, 1):
        print(f"\nProcessing sentence {i}/{num_sentences}: {sentence}")
        
        paraphrases = generator.generate_paraphrases(
            sentence,
            num_paraphrases=args.num_paraphrases
        )
        
        if paraphrases:
            # Add original sentence to each result
            for paraphrase in paraphrases:
                paraphrase['original'] = sentence
                paraphrase['sentence_id'] = i
            
            all_results.extend(paraphrases)
            
            # Display results for this sentence
            display_results(paraphrases, args.show_metrics, generator, sentence)
    
    return all_results


def main():
    """Main CLI function."""
    args = parse_arguments()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    
    # Create generator configuration
    config = ParaphraseConfig(
        model_name=args.model,
        num_return_sequences=args.num_paraphrases,
        temperature=args.temperature,
        top_p=args.top_p,
        max_length=args.max_length
    )
    
    # Initialize generator
    print(f"Initializing paraphrase generator with model: {args.model}")
    generator = ParaphraseGenerator(config)
    
    # Process input based on type
    if args.synthetic:
        all_results = process_synthetic_data(generator, args.synthetic, args)
        
        if args.output:
            save_results(all_results, args.output, args.format)
    
    else:
        # Get input text
        if args.text:
            input_text = args.text
        elif args.file:
            input_text = load_text_from_file(args.file)
        else:
            print("Error: No input text provided")
            sys.exit(1)
        
        # Generate paraphrases
        print(f"Generating {args.num_paraphrases} paraphrases...")
        paraphrases = generator.generate_paraphrases(input_text, args.num_paraphrases)
        
        # Display results
        display_results(paraphrases, args.show_metrics, generator, input_text)
        
        # Save results if requested
        if args.output:
            save_results(paraphrases, args.output, args.format)


if __name__ == "__main__":
    main()
