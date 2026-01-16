"""
Streamlit web interface for the paraphrase generation system.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from paraphrase_generator import ParaphraseGenerator, create_synthetic_dataset
from config import ConfigManager, AppConfig


def setup_page_config():
    """Setup Streamlit page configuration."""
    st.set_page_config(
        page_title="Paraphrase Generation System",
        page_icon="🔄",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def initialize_session_state():
    """Initialize session state variables."""
    if 'generator' not in st.session_state:
        st.session_state.generator = None
    if 'config' not in st.session_state:
        st.session_state.config = None
    if 'paraphrase_history' not in st.session_state:
        st.session_state.paraphrase_history = []


def load_configuration():
    """Load application configuration."""
    config_manager = ConfigManager()
    return config_manager.load_config()


def create_sidebar(config: AppConfig):
    """Create sidebar with configuration options."""
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    model_options = [
        "tuner007/pegasus_paraphrase",
        "t5-small",
        "t5-base",
        "facebook/bart-large-cnn",
        "google/pegasus-large"
    ]
    
    selected_model = st.sidebar.selectbox(
        "Model",
        model_options,
        index=model_options.index(config.model.name)
    )
    
    # Generation parameters
    st.sidebar.subheader("Generation Parameters")
    num_sequences = st.sidebar.slider(
        "Number of paraphrases",
        min_value=1,
        max_value=10,
        value=config.model.num_return_sequences
    )
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=config.model.temperature,
        step=0.1
    )
    
    top_p = st.sidebar.slider(
        "Top-p",
        min_value=0.1,
        max_value=1.0,
        value=config.model.top_p,
        step=0.05
    )
    
    max_length = st.sidebar.slider(
        "Max length",
        min_value=50,
        max_value=200,
        value=config.model.max_length
    )
    
    return {
        'model_name': selected_model,
        'num_return_sequences': num_sequences,
        'temperature': temperature,
        'top_p': top_p,
        'max_length': max_length
    }


def display_main_interface():
    """Display the main interface."""
    st.title("🔄 Paraphrase Generation System")
    st.markdown("Generate high-quality paraphrases using state-of-the-art NLP models")
    
    # Input section
    st.header("📝 Input Text")
    input_text = st.text_area(
        "Enter text to paraphrase:",
        height=100,
        placeholder="Enter your text here...",
        help="The text you want to paraphrase"
    )
    
    return input_text


def generate_paraphrases(generator: ParaphraseGenerator, text: str, params: Dict[str, Any]):
    """Generate paraphrases for the given text."""
    if not text.strip():
        st.warning("Please enter some text to paraphrase.")
        return []
    
    try:
        with st.spinner("Generating paraphrases..."):
            paraphrases = generator.generate_paraphrases(
                text,
                num_paraphrases=params['num_return_sequences']
            )
        
        # Store in history
        st.session_state.paraphrase_history.append({
            'original': text,
            'paraphrases': paraphrases,
            'timestamp': pd.Timestamp.now()
        })
        
        return paraphrases
    
    except Exception as e:
        st.error(f"Error generating paraphrases: {str(e)}")
        return []


def display_paraphrases(paraphrases: List[Dict[str, Any]]):
    """Display generated paraphrases."""
    if not paraphrases:
        return
    
    st.header("✨ Generated Paraphrases")
    
    # Create DataFrame for better display
    df_data = []
    for i, result in enumerate(paraphrases, 1):
        df_data.append({
            'Rank': i,
            'Paraphrase': result['text'],
            'Similarity Score': f"{result['similarity_score']:.3f}"
        })
    
    df = pd.DataFrame(df_data)
    
    # Display table
    st.dataframe(df, use_container_width=True)
    
    # Display individual paraphrases with metrics
    for i, result in enumerate(paraphrases, 1):
        with st.expander(f"Paraphrase {i} (Similarity: {result['similarity_score']:.3f})"):
            st.write(result['text'])
            
            # Similarity score visualization
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['similarity_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Similarity Score"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgray"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)


def display_evaluation_metrics(generator: ParaphraseGenerator, original: str, paraphrases: List[Dict[str, Any]]):
    """Display evaluation metrics."""
    if not paraphrases:
        return
    
    st.header("📊 Evaluation Metrics")
    
    paraphrase_texts = [p['text'] for p in paraphrases]
    metrics = generator.evaluate_paraphrases(original, paraphrase_texts)
    
    if metrics:
        # Create metrics DataFrame
        metrics_df = pd.DataFrame([
            {'Metric': 'ROUGE-1', 'Score': f"{metrics['rouge_1']:.3f}"},
            {'Metric': 'ROUGE-2', 'Score': f"{metrics['rouge_2']:.3f}"},
            {'Metric': 'ROUGE-L', 'Score': f"{metrics['rouge_l']:.3f}"},
            {'Metric': 'BLEU (Mean)', 'Score': f"{metrics['bleu_mean']:.3f}"},
            {'Metric': 'BLEU (Std)', 'Score': f"{metrics['bleu_std']:.3f}"}
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(metrics_df, use_container_width=True)
        
        with col2:
            # Create bar chart for metrics
            fig = px.bar(
                metrics_df,
                x='Metric',
                y='Score',
                title='Evaluation Metrics',
                color='Score',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def display_history():
    """Display paraphrase history."""
    if not st.session_state.paraphrase_history:
        return
    
    st.header("📚 History")
    
    # Create history DataFrame
    history_data = []
    for i, entry in enumerate(st.session_state.paraphrase_history):
        history_data.append({
            'Index': i + 1,
            'Original': entry['original'][:50] + "..." if len(entry['original']) > 50 else entry['original'],
            'Paraphrases': len(entry['paraphrases']),
            'Timestamp': entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        })
    
    history_df = pd.DataFrame(history_data)
    
    # Display history
    selected_indices = st.multiselect(
        "Select entries to view:",
        range(len(history_data)),
        format_func=lambda x: f"Entry {x + 1}: {history_data[x]['Original']}"
    )
    
    if selected_indices:
        for idx in selected_indices:
            entry = st.session_state.paraphrase_history[idx]
            with st.expander(f"Entry {idx + 1} - {entry['timestamp'].strftime('%H:%M:%S')}"):
                st.write("**Original:**", entry['original'])
                st.write("**Paraphrases:**")
                for j, paraphrase in enumerate(entry['paraphrases'], 1):
                    st.write(f"{j}. {paraphrase['text']} (Similarity: {paraphrase['similarity_score']:.3f})")


def display_synthetic_data_demo():
    """Display synthetic data generation demo."""
    st.header("🧪 Synthetic Data Demo")
    
    if st.button("Generate Sample Sentences"):
        with st.spinner("Generating synthetic sentences..."):
            sentences = create_synthetic_dataset(10)
        
        st.write("**Sample Sentences:**")
        for i, sentence in enumerate(sentences, 1):
            st.write(f"{i}. {sentence}")
        
        # Allow user to select sentences for paraphrasing
        selected_sentence = st.selectbox(
            "Select a sentence to paraphrase:",
            sentences
        )
        
        if st.button("Paraphrase Selected Sentence"):
            return selected_sentence
    
    return None


def main():
    """Main application function."""
    setup_page_config()
    initialize_session_state()
    
    # Load configuration
    config = load_configuration()
    st.session_state.config = config
    
    # Create sidebar
    params = create_sidebar(config)
    
    # Initialize generator if needed
    if st.session_state.generator is None or st.session_state.generator.config.model_name != params['model_name']:
        from paraphrase_generator import ParaphraseConfig
        generator_config = ParaphraseConfig(
            model_name=params['model_name'],
            num_return_sequences=params['num_return_sequences'],
            temperature=params['temperature'],
            top_p=params['top_p'],
            max_length=params['max_length']
        )
        st.session_state.generator = ParaphraseGenerator(generator_config)
    
    # Main interface
    input_text = display_main_interface()
    
    # Synthetic data demo
    synthetic_text = display_synthetic_data_demo()
    if synthetic_text:
        input_text = synthetic_text
    
    # Generate paraphrases
    if st.button("🔄 Generate Paraphrases", type="primary"):
        paraphrases = generate_paraphrases(st.session_state.generator, input_text, params)
        
        if paraphrases:
            display_paraphrases(paraphrases)
            
            if config.ui.show_evaluation_metrics:
                display_evaluation_metrics(st.session_state.generator, input_text, paraphrases)
    
    # Display history
    display_history()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, Transformers, and Hugging Face. "
        "For more information, check out the [GitHub repository](https://github.com/your-repo)."
    )


if __name__ == "__main__":
    main()
