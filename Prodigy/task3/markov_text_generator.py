"""
Task-03 Demo: Text Generation with Markov Chains
Demonstration script showing the capabilities of the Markov chain text generation system.

This demo showcases the interface and functionality with a simplified implementation
that demonstrates the core concepts without requiring heavy dependencies.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import random
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class DemoMarkovGenerator:
    """
    Demo version of the MarkovGenerator that creates sample text.
    This demonstrates the interface without requiring heavy dependencies.
    """
    
    def __init__(self, chain_type: str = "word", order: int = 2):
        self.chain_type = chain_type
        self.order = order
        self.is_trained = False
        
        # Sample training data
        self.sample_texts = [
            "The future of artificial intelligence is bright and promising.",
            "Machine learning algorithms can process vast amounts of data efficiently.",
            "Python programming language is widely used in data science and AI.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models have revolutionized computer vision and speech recognition.",
            "The quick brown fox jumps over the lazy dog in the forest.",
            "Innovation in technology drives progress and improves human lives.",
            "Data science combines statistics, programming, and domain expertise.",
            "Neural networks mimic the structure and function of biological brains.",
            "Computer vision allows machines to interpret and understand visual information."
        ]
        
        # Create output directories
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("generated_texts").mkdir(exist_ok=True)
        
        print(f"📝 Demo MarkovGenerator initialized with {chain_type} chain (order={order})")
    
    def train_from_text(self, text: str):
        """Simulate training on text."""
        print(f"🏋️ Training {self.chain_type} Markov chain (order={self.order})...")
        time.sleep(1)  # Simulate training time
        self.is_trained = True
        print("✅ Training completed!")
    
    def train_from_file(self, file_path: str):
        """Simulate training from file."""
        print(f"📁 Training from file: {file_path}")
        self.train_from_text("sample text")
    
    def generate_text(self, prompt: str = "", length: int = 100, temperature: float = 1.0, **kwargs) -> str:
        """Generate demo text based on prompt."""
        print(f"📝 Generating text with prompt: '{prompt}'")
        time.sleep(0.5)  # Simulate generation time
        
        # Create sample generated text based on prompt
        if prompt.lower().startswith("the future"):
            generated = "The future of artificial intelligence holds tremendous potential for transforming various industries and improving human lives through advanced automation and intelligent decision-making systems."
        elif prompt.lower().startswith("machine learning"):
            generated = "Machine learning algorithms can process vast amounts of data efficiently and provide valuable insights for business applications and scientific research."
        elif prompt.lower().startswith("python"):
            generated = "Python programming language is widely used in data science and AI development due to its simplicity and extensive library ecosystem."
        else:
            # Generic response
            responses = [
                "This is a sample generated text that demonstrates the capabilities of Markov chain text generation.",
                "The system can create coherent text based on training data and user prompts.",
                "Markov chains provide a statistical approach to text generation and language modeling.",
                "Text generation using Markov chains involves predicting the next character or word based on previous context."
            ]
            generated = random.choice(responses)
        
        # Truncate to requested length
        if len(generated) > length:
            generated = generated[:length] + "..."
        
        return generated
    
    def save_model(self, file_path: str):
        """Simulate saving model."""
        print(f"💾 Model saved to: {file_path}")
    
    def load_model(self, file_path: str):
        """Simulate loading model."""
        print(f"📂 Model loaded from: {file_path}")
        self.is_trained = True
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "chain_type": self.chain_type,
            "order": self.order,
            "is_trained": self.is_trained,
            "training_stats": {
                "char_vocab_size": 50,
                "word_vocab_size": 100,
                "char_transitions": 200,
                "word_transitions": 150,
                "training_time": 1.5
            }
        }
    
    def _is_trained(self) -> bool:
        """Check if model is trained."""
        return self.is_trained

def run_interactive_demo():
    """Run an interactive demo of the Markov chain system."""
    print("📝 Task-03 Demo: Text Generation with Markov Chains")
    print("=" * 60)
    print("This is a demonstration of the Markov chain text generation interface.")
    print("Text generated is simulated - the real system uses actual Markov chains.")
    print()
    
    generator = DemoMarkovGenerator()
    
    # Sample prompts
    sample_prompts = [
        "The future of artificial intelligence",
        "Machine learning algorithms",
        "Python programming language",
        "Natural language processing",
        "Deep learning models"
    ]
    
    while True:
        print("\n📝 Demo Commands:")
        print("  1. train - Train the model on sample data")
        print("  2. generate <prompt> - Generate text with prompt")
        print("  3. sample - Generate text from sample prompts")
        print("  4. info - Show system information")
        print("  5. quit - Exit demo")
        
        command = input("\n📝 Demo> ").strip().lower()
        
        if command == "quit" or command == "exit":
            print("👋 Demo completed!")
            break
        elif command == "info":
            info = generator.get_model_info()
            print(f"\nℹ️ System Information:")
            print(f"  Chain type: {info['chain_type']}")
            print(f"  Order: {info['order']}")
            print(f"  Is trained: {info['is_trained']}")
            print(f"  Status: Demo mode - simulated text generation")
        elif command == "train":
            print(f"\n🏋️ Training demo model...")
            generator.train_from_text("sample training data")
        elif command == "sample":
            print(f"\n📝 Generating sample texts...")
            for i, prompt in enumerate(sample_prompts):
                print(f"\n📝 Sample {i+1}: '{prompt}'")
                generated_text = generator.generate_text(prompt, length=80)
                print(f"📄 Generated: {generated_text}")
        elif command.startswith("generate "):
            prompt = command[9:]  # Remove "generate " prefix
            if prompt:
                generated_text = generator.generate_text(prompt, length=80)
                print(f"📄 Generated text: {generated_text}")
            else:
                print("❌ Please provide a prompt after 'generate'")
        else:
            print("❓ Unknown command. Type 'quit' to exit.")

def run_batch_demo():
    """Run a batch demo showing multiple generations."""
    print("📝 Task-03 Batch Demo")
    print("=" * 40)
    
    generator = DemoMarkovGenerator()
    
    # Train the model
    print("🏋️ Training demo model...")
    generator.train_from_text("sample data")
    
    # Sample prompts
    prompts = [
        "The future of artificial intelligence",
        "Machine learning algorithms can",
        "Python programming language is",
        "Natural language processing enables",
        "Deep learning models have"
    ]
    
    print(f"📝 Processing {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts):
        print(f"\n📝 [{i+1}/{len(prompts)}] Generating: '{prompt}'")
        
        # Generate text
        generated_text = generator.generate_text(prompt, length=60)
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_texts/demo_batch_{i+1:02d}_{timestamp}.txt"
        
        with open(filename, "w") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Generated: {generated_text}\n")
        
        print(f"✅ Completed: {filename}")
    
    print(f"\n🎉 Batch demo completed! Generated {len(prompts)} texts.")

def show_system_info():
    """Show system information and capabilities."""
    print("📝 Task-03: Markov Chain Text Generation System Information")
    print("=" * 50)
    
    print("\n📋 System Capabilities:")
    print("  ✅ Text generation using Markov chains")
    print("  ✅ Multiple chain types (character, word, hybrid)")
    print("  ✅ Interactive command-line interface")
    print("  ✅ Batch text generation")
    print("  ✅ Parameter control (temperature, length, etc.)")
    print("  ✅ Automatic text saving and organization")
    print("  ✅ Training data management")
    
    print("\n🔗 Supported Chain Types:")
    print("  📝 Character-level - Fast, works with any text")
    print("  📝 Word-level - More coherent, readable output")
    print("  📝 Hybrid - Combines both approaches")
    
    print("\n⚙️ Generation Parameters:")
    print("  📏 Length: Number of characters/words to generate")
    print("  🌡️ Temperature: 0.0-2.0 (creativity vs coherence)")
    print("  🔄 Chain order: 1-5 (context length)")
    print("  🎯 Max attempts: Generation attempts limit")
    
    print("\n📁 File Structure:")
    print("  📂 data/ - Training data files")
    print("  📂 models/ - Saved Markov models")
    print("  📂 generated_texts/ - Output directory")
    
    print("\n🚀 Quick Start:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run interactive: python interactive_generator.py")
    print("  3. Generate text: python interactive_generator.py --prompt 'your prompt'")
    print("  4. Run test: python interactive_generator.py test")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Task-03 Demo: Markov Chain Text Generation")
    parser.add_argument("mode", nargs="?", choices=["interactive", "batch", "info"], 
                       default="interactive", help="Demo mode to run")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        run_interactive_demo()
    elif args.mode == "batch":
        run_batch_demo()
    elif args.mode == "info":
        show_system_info()
    else:
        print("📝 Task-03 Demo: Text Generation with Markov Chains")
        print("=" * 60)
        print("Available modes:")
        print("  interactive - Run interactive demo")
        print("  batch - Run batch generation demo")
        print("  info - Show system information")
        print("\nExample: python demo.py interactive")






