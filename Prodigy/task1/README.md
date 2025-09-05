# Task-01: Text Generation with GPT-2

This project implements a complete text generation system using GPT-2, a transformer model developed by OpenAI. The system can fine-tune the model on custom datasets and generate coherent, contextually relevant text based on given prompts.

## 🎯 Project Overview

The goal of this task is to:
- Train a model to generate coherent and contextually relevant text
- Fine-tune GPT-2 on a custom dataset
- Create text that mimics the style and structure of training data
- Provide an interactive interface for text generation

## 🚀 Features

- **Fine-tuning**: Adapt GPT-2 to your specific domain or style
- **Text Generation**: Generate text from custom prompts
- **Interactive Interface**: User-friendly command-line interface
- **Model Evaluation**: Assess generation quality with metrics
- **Customizable Parameters**: Control temperature, top-k, top-p, and more
- **Multiple Outputs**: Generate multiple sequences from the same prompt

## 📋 Requirements

- Python 3.7+
- PyTorch
- Transformers library
- CUDA (optional, for GPU acceleration)

## 🛠️ Installation

1. **Clone or download this project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python interactive_generator.py test
   ```

## 📁 Project Structure

```
├── gpt2_text_generator.py      # Main GPT-2 implementation
├── interactive_generator.py    # Interactive interface
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   └── sample_texts.txt        # Sample training data
├── fine_tuned_model/           # Fine-tuned model (created after training)
└── processed_data/             # Processed training data
```

## 🎮 Usage

### Quick Start

1. **Run the interactive interface**:
   ```bash
   python interactive_generator.py
   ```

2. **Generate text**:
   ```
   generate The future of artificial intelligence
   ```

3. **Adjust settings**:
   ```
   settings
   ```

### Programmatic Usage

```python
from gpt2_text_generator import GPT2TextGenerator

# Initialize the generator
generator = GPT2TextGenerator()

# Generate text
prompt = "The future of artificial intelligence"
generated_texts = generator.generate_text(
    prompt=prompt,
    max_length=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9
)

print(generated_texts[0])
```

### Fine-tuning on Custom Data

1. **Prepare your training data** in `data/sample_texts.txt` (one sentence per line)

2. **Run the main script**:
   ```bash
   python gpt2_text_generator.py
   ```

3. **The script will automatically**:
   - Load the pre-trained GPT-2 model
   - Prepare and tokenize your training data
   - Fine-tune the model for 2 epochs
   - Save the fine-tuned model
   - Generate sample text

## ⚙️ Configuration

### Generation Parameters

- **max_length**: Maximum number of tokens to generate (default: 100)
- **temperature**: Controls randomness (0.0 = deterministic, 1.0 = very random)
- **top_k**: Number of highest probability tokens to consider
- **top_p**: Cumulative probability threshold for token selection
- **num_return_sequences**: Number of different sequences to generate

### Training Parameters

- **num_epochs**: Number of training epochs (default: 2)
- **batch_size**: Training batch size (default: 2)
- **learning_rate**: Learning rate for fine-tuning (default: 5e-5)

## 📊 Model Evaluation

The system includes evaluation metrics:

- **Text Length**: Average number of words generated
- **Perplexity**: Measure of how well the model predicts the next token
- **Coherence**: Qualitative assessment of generated text

## 🔧 Customization

### Adding Custom Training Data

1. Replace or extend `data/sample_texts.txt` with your own text
2. Each line should be a complete sentence or paragraph
3. Ensure the text style matches your desired output

### Modifying Model Parameters

Edit the parameters in `gpt2_text_generator.py`:

```python
# In the fine_tune method
generator.fine_tune(
    dataset, 
    num_epochs=5,        # More epochs for better results
    batch_size=4,        # Larger batch size if you have more memory
    learning_rate=3e-5   # Lower learning rate for more stable training
)
```

### Using Different GPT-2 Variants

```python
# Use GPT-2 Medium (355M parameters)
generator = GPT2TextGenerator(model_name="gpt2-medium")

# Use GPT-2 Large (774M parameters)
generator = GPT2TextGenerator(model_name="gpt2-large")
```

## 🎯 Example Prompts

Try these example prompts to test the system:

- "The future of artificial intelligence"
- "Machine learning is"
- "Python programming"
- "Natural language processing"
- "Deep learning models"
- "The quick brown fox"
- "Innovation in technology"

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce batch_size in fine-tuning
   - Use smaller model variant (gpt2 instead of gpt2-medium)
   - Close other applications to free memory

2. **Slow Generation**:
   - Use GPU if available (CUDA)
   - Reduce max_length parameter
   - Use greedy decoding (set do_sample=False)

3. **Poor Quality Output**:
   - Increase training epochs
   - Add more diverse training data
   - Adjust temperature and top-k parameters

### Performance Tips

- **GPU Usage**: The model automatically uses CUDA if available
- **Batch Processing**: Generate multiple sequences at once for efficiency
- **Model Caching**: Fine-tuned models are saved and reused automatically

## 📚 Technical Details

### Architecture

- **Base Model**: GPT-2 (124M parameters)
- **Tokenization**: GPT-2 tokenizer with byte-pair encoding
- **Training**: Language modeling with causal attention
- **Generation**: Autoregressive text generation

### Training Process

1. **Data Preparation**: Text is tokenized and split into sequences
2. **Fine-tuning**: Model is trained on custom data using language modeling loss
3. **Validation**: Model performance is evaluated on test prompts
4. **Saving**: Fine-tuned model and tokenizer are saved locally

## 🤝 Contributing

Feel free to extend this project:

- Add support for other language models
- Implement additional evaluation metrics
- Create a web interface
- Add support for different text formats
- Implement model compression techniques

## 📄 License

This project is for educational purposes. Please respect the licenses of the underlying libraries (PyTorch, Transformers, etc.).

## 🎉 Success!

You've successfully completed Task-01: Text Generation with GPT-2! The system can now:

✅ Load and fine-tune GPT-2 models  
✅ Generate coherent text from prompts  
✅ Evaluate model performance  
✅ Provide an interactive interface  
✅ Handle custom training data  

Happy text generating! 🚀
