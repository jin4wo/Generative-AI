# Task-03: Text Generation with Markov Chains

This project implements a complete text generation system using Markov chains. The system can learn from training data and generate coherent text by predicting the probability of characters or words based on previous sequences, providing an interactive interface for creative text generation.

## 🎯 Project Overview

The goal of this task is to:
- Implement a statistical model using Markov chains for text generation
- Predict character/word probabilities based on previous sequences
- Create coherent text that mimics the style of training data
- Provide an interactive interface for text generation
- Support both character-level and word-level Markov chains

## 🚀 Features

- **Multiple Chain Types**: Support for character-level and word-level Markov chains
- **Text Generation**: Create coherent text from training data
- **Interactive Interface**: User-friendly command-line interface
- **Training Data Management**: Load and process custom training data
- **Chain Order Control**: Adjustable context length for predictions
- **Text Quality Control**: Configurable generation parameters
- **Batch Generation**: Generate multiple text sequences
- **Model Persistence**: Save and load trained models

## 📋 Requirements

- Python 3.7+
- NumPy for numerical operations
- Collections for data structures
- Random for text generation
- JSON for model persistence
- Pathlib for file operations

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
├── markov_generator.py         # Main Markov chain implementation
├── interactive_generator.py    # Interactive interface
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Training data files
├── models/                     # Saved Markov models
└── generated_texts/            # Generated text outputs
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
from markov_generator import MarkovGenerator

# Initialize the generator
generator = MarkovGenerator(chain_type="word", order=2)

# Train on data
generator.train_from_file("data/sample_texts.txt")

# Generate text
prompt = "The future of artificial intelligence"
generated_text = generator.generate_text(
    prompt=prompt,
    length=100,
    temperature=1.0
)

print(generated_text)
```

### Using Different Chain Types

```python
# Character-level Markov chain
generator = MarkovGenerator(chain_type="character", order=3)

# Word-level Markov chain
generator = MarkovGenerator(chain_type="word", order=2)

# Mixed approach
generator = MarkovGenerator(chain_type="hybrid", order=2)
```

## ⚙️ Configuration

### Generation Parameters

- **length**: Number of characters/words to generate (default: 100)
- **temperature**: Controls randomness (0.0 = deterministic, 2.0 = very random)
- **max_attempts**: Maximum attempts for valid generation (default: 1000)
- **start_with_prompt**: Whether to start generation with the given prompt

### Model Parameters

- **chain_type**: "character", "word", or "hybrid"
- **order**: Context length for predictions (1-5 recommended)
- **smoothing**: Laplace smoothing factor (default: 0.1)

## 📊 Model Comparison

### Character-Level Markov Chains
- **Pros**: Fast training, works with any text, captures character patterns
- **Cons**: Less coherent at word boundaries, may generate gibberish
- **Best for**: Short sequences, character-based analysis, quick prototypes

### Word-Level Markov Chains
- **Pros**: More coherent text, better word relationships, readable output
- **Cons**: Requires more training data, slower training
- **Best for**: Longer texts, readable content, word-based analysis

### Hybrid Approach
- **Pros**: Combines benefits of both approaches
- **Cons**: More complex implementation, higher computational cost
- **Best for**: Advanced text generation, mixed content types

## 🔧 Customization

### Adding Custom Training Data

1. Place your text files in the `data/` directory
2. Use the `load_data` command in the interactive interface
3. Ensure text is properly formatted (one sentence per line recommended)

### Modifying Generation Parameters

Edit the parameters in `markov_generator.py`:

```python
# In the generate_text method
text = generator.generate_text(
    prompt=prompt,
    length=200,           # Generate 200 characters/words
    temperature=1.5,      # Higher randomness
    max_attempts=2000     # More attempts for valid generation
)
```

### Using Different Chain Orders

```python
# First-order Markov chain (simple)
generator = MarkovGenerator(chain_type="word", order=1)

# Second-order Markov chain (recommended)
generator = MarkovGenerator(chain_type="word", order=2)

# Higher-order Markov chain (more context)
generator = MarkovGenerator(chain_type="word", order=3)
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

1. **Poor Quality Output**:
   - Increase training data size
   - Adjust chain order (try order=2 or 3)
   - Reduce temperature for more coherent text
   - Use word-level instead of character-level

2. **Slow Generation**:
   - Reduce chain order
   - Use character-level chains
   - Limit generation length
   - Optimize training data size

3. **Repetitive Output**:
   - Increase temperature
   - Add more training data variety
   - Try different chain orders
   - Use hybrid approach

### Performance Tips

- **Training Data**: Use diverse, high-quality text for better results
- **Chain Order**: Balance between context and performance
- **Temperature**: Adjust based on desired creativity vs coherence
- **Model Persistence**: Save trained models to avoid retraining

## 📚 Technical Details

### Architecture

- **Markov Chain**: Statistical model for sequence prediction
- **Transition Matrix**: Stores probability distributions
- **Context Management**: Handles variable-length contexts
- **Text Processing**: Tokenization and normalization

### Training Process

1. **Data Loading**: Read and preprocess training text
2. **Context Extraction**: Build transition probabilities
3. **Matrix Construction**: Create probability transition matrix
4. **Model Validation**: Verify model quality and coverage

### Generation Process

1. **Prompt Processing**: Handle initial context
2. **Probability Sampling**: Select next character/word
3. **Context Update**: Maintain sliding context window
4. **Output Validation**: Ensure generated text quality

## 🤝 Contributing

Feel free to extend this project:

- Add support for other Markov chain variants
- Implement additional text processing features
- Create a web interface
- Add support for different languages
- Implement advanced smoothing techniques

## 📄 License

This project is for educational purposes. Please respect the licenses of the underlying libraries.

## 🎉 Success!

You've successfully completed Task-03: Text Generation with Markov Chains! The system can now:

✅ Build Markov chain models from training data  
✅ Generate coherent text using statistical predictions  
✅ Provide an interactive interface  
✅ Control generation parameters  
✅ Handle different chain types and orders  

Happy text generating! 📝

