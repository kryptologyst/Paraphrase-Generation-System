"""
Test suite for the paraphrase generation system.
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from paraphrase_generator import ParaphraseGenerator, ParaphraseConfig, create_synthetic_dataset
from config import ConfigManager, AppConfig, ModelConfig, UIConfig, LoggingConfig


class TestParaphraseConfig(unittest.TestCase):
    """Test ParaphraseConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ParaphraseConfig()
        
        self.assertEqual(config.model_name, "tuner007/pegasus_paraphrase")
        self.assertEqual(config.max_length, 100)
        self.assertEqual(config.num_return_sequences, 3)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.9)
        self.assertEqual(config.top_k, 50)
        self.assertTrue(config.do_sample)
        self.assertEqual(config.device, "auto")
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ParaphraseConfig(
            model_name="t5-small",
            max_length=50,
            num_return_sequences=5,
            temperature=0.5
        )
        
        self.assertEqual(config.model_name, "t5-small")
        self.assertEqual(config.max_length, 50)
        self.assertEqual(config.num_return_sequences, 5)
        self.assertEqual(config.temperature, 0.5)


class TestParaphraseGenerator(unittest.TestCase):
    """Test ParaphraseGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = ParaphraseConfig(
            model_name="t5-small",
            num_return_sequences=2
        )
    
    @patch('paraphrase_generator.AutoTokenizer')
    @patch('paraphrase_generator.T5ForConditionalGeneration')
    def test_initialization(self, mock_model, mock_tokenizer):
        """Test generator initialization."""
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        
        generator = ParaphraseGenerator(self.config)
        
        self.assertEqual(generator.config, self.config)
        self.assertIsNotNone(generator.model)
        self.assertIsNotNone(generator.tokenizer)
    
    def test_calculate_similarity(self):
        """Test similarity calculation."""
        generator = ParaphraseGenerator(self.config)
        
        # Test identical sentences
        similarity = generator._calculate_similarity("Hello world", "Hello world")
        self.assertEqual(similarity, 1.0)
        
        # Test completely different sentences
        similarity = generator._calculate_similarity("Hello world", "Goodbye universe")
        self.assertEqual(similarity, 0.0)
        
        # Test partial overlap
        similarity = generator._calculate_similarity("Hello world", "Hello universe")
        self.assertGreater(similarity, 0.0)
        self.assertLess(similarity, 1.0)
    
    def test_empty_similarity(self):
        """Test similarity calculation with empty strings."""
        generator = ParaphraseGenerator(self.config)
        
        similarity = generator._calculate_similarity("", "Hello world")
        self.assertEqual(similarity, 0.0)
        
        similarity = generator._calculate_similarity("Hello world", "")
        self.assertEqual(similarity, 0.0)
        
        similarity = generator._calculate_similarity("", "")
        self.assertEqual(similarity, 0.0)


class TestSyntheticDataset(unittest.TestCase):
    """Test synthetic dataset generation."""
    
    def test_dataset_size(self):
        """Test that dataset has correct size."""
        dataset = create_synthetic_dataset(10)
        self.assertEqual(len(dataset), 10)
    
    def test_dataset_content(self):
        """Test that dataset contains valid sentences."""
        dataset = create_synthetic_dataset(5)
        
        for sentence in dataset:
            self.assertIsInstance(sentence, str)
            self.assertGreater(len(sentence), 0)
            self.assertTrue(sentence.endswith('.'))
    
    def test_dataset_variety(self):
        """Test that dataset has variety."""
        dataset = create_synthetic_dataset(20)
        
        # Check that we have different sentences
        unique_sentences = set(dataset)
        self.assertGreater(len(unique_sentences), 1)


class TestConfigManager(unittest.TestCase):
    """Test ConfigManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_default_config(self):
        """Test default configuration creation."""
        config_manager = ConfigManager(self.config_path)
        config = config_manager._get_default_config()
        
        self.assertIsInstance(config, AppConfig)
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.ui, UIConfig)
        self.assertIsInstance(config.logging, LoggingConfig)
    
    def test_config_save_load(self):
        """Test configuration save and load."""
        config_manager = ConfigManager(self.config_path)
        
        # Create and save config
        config = config_manager._get_default_config()
        config_manager.save_config(config)
        
        # Load config
        loaded_config = config_manager.load_config()
        
        self.assertEqual(config.model.name, loaded_config.model.name)
        self.assertEqual(config.ui.title, loaded_config.ui.title)
        self.assertEqual(config.logging.level, loaded_config.logging.level)


class TestAppConfig(unittest.TestCase):
    """Test AppConfig class."""
    
    def test_from_dict(self):
        """Test AppConfig creation from dictionary."""
        config_dict = {
            'model': {
                'name': 't5-small',
                'max_length': 50
            },
            'ui': {
                'title': 'Test App'
            },
            'logging': {
                'level': 'DEBUG'
            }
        }
        
        config = AppConfig.from_dict(config_dict)
        
        self.assertEqual(config.model.name, 't5-small')
        self.assertEqual(config.model.max_length, 50)
        self.assertEqual(config.ui.title, 'Test App')
        self.assertEqual(config.logging.level, 'DEBUG')
    
    def test_to_dict(self):
        """Test AppConfig conversion to dictionary."""
        config = AppConfig(
            model=ModelConfig(name='t5-small'),
            ui=UIConfig(title='Test App'),
            logging=LoggingConfig(level='DEBUG')
        )
        
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict['model']['name'], 't5-small')
        self.assertEqual(config_dict['ui']['title'], 'Test App')
        self.assertEqual(config_dict['logging']['level'], 'DEBUG')


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    @patch('paraphrase_generator.AutoTokenizer')
    @patch('paraphrase_generator.T5ForConditionalGeneration')
    def test_end_to_end_workflow(self, mock_model, mock_tokenizer):
        """Test end-to-end workflow."""
        # Mock the model and tokenizer
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()
        
        # Create generator
        config = ParaphraseConfig(model_name="t5-small")
        generator = ParaphraseGenerator(config)
        
        # Test synthetic dataset creation
        dataset = create_synthetic_dataset(3)
        self.assertEqual(len(dataset), 3)
        
        # Test configuration management
        config_manager = ConfigManager()
        app_config = config_manager._get_default_config()
        self.assertIsInstance(app_config, AppConfig)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
