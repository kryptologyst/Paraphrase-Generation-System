#!/usr/bin/env python3
"""
Quick test script to verify the system works without downloading models.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    try:
        from paraphrase_generator import ParaphraseConfig, create_synthetic_dataset
        from config import ConfigManager, AppConfig
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_synthetic_data():
    """Test synthetic data generation."""
    try:
        from paraphrase_generator import create_synthetic_dataset
        sentences = create_synthetic_dataset(3)
        assert len(sentences) == 3
        assert all(isinstance(s, str) for s in sentences)
        print("✅ Synthetic data generation works")
        return True
    except Exception as e:
        print(f"❌ Synthetic data error: {e}")
        return False

def test_config():
    """Test configuration management."""
    try:
        from config import ConfigManager, AppConfig
        config_manager = ConfigManager()
        config = config_manager._get_default_config()
        assert isinstance(config, AppConfig)
        print("✅ Configuration management works")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_similarity_calculation():
    """Test similarity calculation without model."""
    try:
        from paraphrase_generator import ParaphraseGenerator, ParaphraseConfig
        
        # Create a mock generator (won't load actual model)
        config = ParaphraseConfig()
        
        # Test similarity calculation directly
        generator = ParaphraseGenerator.__new__(ParaphraseGenerator)
        generator.config = config
        
        similarity = generator._calculate_similarity("Hello world", "Hello universe")
        assert 0 <= similarity <= 1
        print("✅ Similarity calculation works")
        return True
    except Exception as e:
        print(f"❌ Similarity calculation error: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Paraphrase Generation System...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_synthetic_data,
        test_config,
        test_similarity_calculation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run the web app: streamlit run web_app/app.py")
        print("3. Try the CLI: python src/cli.py 'Hello world'")
        print("4. Run the example: python example.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()
