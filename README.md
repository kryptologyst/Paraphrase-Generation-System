# Paraphrase Generation System

State-of-the-art paraphrase generation system built with Hugging Face Transformers, featuring multiple interfaces and advanced evaluation metrics.

## Features

- **Multiple Model Support**: Pegasus, T5, BART, and other transformer models
- **Advanced Sampling**: Temperature, top-p, and top-k sampling for diverse outputs
- **Evaluation Metrics**: ROUGE, BLEU, and semantic similarity scoring
- **Multiple Interfaces**: CLI, Streamlit web app, and Python API
- **Synthetic Data Generation**: Built-in dataset creation for testing
- **Configuration Management**: YAML/JSON configuration support
- **Comprehensive Testing**: Unit tests and integration tests
- **Modern Architecture**: Type hints, logging, and clean code structure

## 📁 Project Structure

```
├── src/                          # Source code
│   ├── paraphrase_generator.py   # Core paraphrase generation logic
│   ├── config.py                 # Configuration management
│   └── cli.py                    # Command-line interface
├── web_app/                      # Web interface
│   └── app.py                    # Streamlit application
├── tests/                        # Test suite
│   └── test_paraphrase_generator.py
├── config/                       # Configuration files
├── data/                         # Data storage
├── models/                       # Model storage
├── requirements.txt              # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Paraphrase-Generation-System.git
   cd Paraphrase-Generation-System
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Web Interface (Recommended)

Launch the Streamlit web application:

```bash
streamlit run web_app/app.py
```

Open your browser to `http://localhost:8501` and start generating paraphrases!

### Command Line Interface

Generate paraphrases from the command line:

```bash
# Basic usage
python src/cli.py "The quick brown fox jumps over the lazy dog."

# With custom parameters
python src/cli.py "Hello world" --num-paraphrases 5 --temperature 0.8 --model t5-small

# From file
python src/cli.py --file input.txt --output results.json --format json

# Generate synthetic data
python src/cli.py --synthetic 10 --show-metrics
```

### Python API

Use the system programmatically:

```python
from src.paraphrase_generator import ParaphraseGenerator, ParaphraseConfig

# Create configuration
config = ParaphraseConfig(
    model_name="tuner007/pegasus_paraphrase",
    num_return_sequences=3,
    temperature=0.7
)

# Initialize generator
generator = ParaphraseGenerator(config)

# Generate paraphrases
text = "The quick brown fox jumps over the lazy dog."
paraphrases = generator.generate_paraphrases(text)

for i, result in enumerate(paraphrases, 1):
    print(f"Paraphrase {i}: {result['text']}")
    print(f"Similarity: {result['similarity_score']:.3f}")
```

## Supported Models

| Model | Description | Best For |
|-------|-------------|----------|
| `tuner007/pegasus_paraphrase` | Fine-tuned Pegasus for paraphrasing | High-quality paraphrases |
| `t5-small` | Small T5 model | Fast inference |
| `t5-base` | Base T5 model | Balanced quality/speed |
| `facebook/bart-large-cnn` | BART model | Abstractive paraphrasing |
| `google/pegasus-large` | Large Pegasus model | Best quality (slower) |

## Configuration

The system uses YAML configuration files. Create `config/config.yaml`:

```yaml
model:
  name: "tuner007/pegasus_paraphrase"
  max_length: 100
  num_return_sequences: 3
  temperature: 0.7
  top_p: 0.9
  top_k: 50
  do_sample: true
  device: "auto"

ui:
  title: "Paraphrase Generation System"
  description: "Generate high-quality paraphrases using state-of-the-art NLP models"
  max_input_length: 500
  default_num_paraphrases: 3
  show_similarity_scores: true
  show_evaluation_metrics: true

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file_path: null
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test
python -m pytest tests/test_paraphrase_generator.py::TestParaphraseGenerator -v
```

## Evaluation Metrics

The system provides comprehensive evaluation metrics:

- **ROUGE Scores**: Measures overlap between original and paraphrased text
- **BLEU Scores**: Measures translation quality
- **Semantic Similarity**: Word overlap-based similarity scoring
- **Custom Metrics**: Extensible metric system

## 🔧 Advanced Usage

### Batch Processing

Process multiple texts at once:

```python
texts = [
    "The weather is beautiful today.",
    "Machine learning is transforming industries.",
    "Python is a versatile programming language."
]

results = generator.batch_paraphrase(texts, num_paraphrases=2)
```

### Custom Evaluation

Evaluate paraphrases with custom metrics:

```python
original = "The quick brown fox jumps over the lazy dog."
paraphrases = ["A fast brown fox leaps over the lazy dog."]

metrics = generator.evaluate_paraphrases(original, paraphrases)
print(f"ROUGE-1: {metrics['rouge_1']:.3f}")
print(f"BLEU: {metrics['bleu_mean']:.3f}")
```

### Synthetic Data Generation

Generate test datasets:

```python
from src.paraphrase_generator import create_synthetic_dataset

# Generate 100 synthetic sentences
sentences = create_synthetic_dataset(100)
```

## Performance Optimization

### GPU Acceleration

The system automatically detects and uses available GPUs:

- **CUDA**: NVIDIA GPUs
- **MPS**: Apple Silicon (M1/M2) GPUs
- **CPU**: Fallback for all systems

### Model Caching

Models are cached automatically by Hugging Face transformers. First-time downloads may take a few minutes.

### Memory Optimization

For large-scale processing:

```python
config = ParaphraseConfig(
    model_name="t5-small",  # Use smaller models
    max_length=50,          # Limit output length
    device="cpu"            # Force CPU if memory limited
)
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `python -m pytest tests/`
6. Commit your changes: `git commit -am 'Add feature'`
7. Push to the branch: `git push origin feature-name`
8. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the transformers library
- [Streamlit](https://streamlit.io/) for the web interface framework
- The open-source NLP community for model contributions

## Support

- **Issues**: Report bugs and request features on GitHub Issues
- **Discussions**: Join community discussions on GitHub Discussions
- **Documentation**: Check the inline documentation and examples

## Future Enhancements

- [ ] Support for more languages
- [ ] Fine-tuning capabilities
- [ ] Advanced semantic similarity using sentence embeddings
- [ ] Real-time collaboration features
- [ ] Model comparison dashboard
- [ ] API endpoint for integration
- [ ] Docker containerization
- [ ] Kubernetes deployment guides

# Paraphrase-Generation-System
