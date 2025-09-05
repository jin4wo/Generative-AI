"""
Interactive Markov Chain Text Generator Interface
Task-03: Command-line interface for Markov chain text generation

This module provides an interactive command-line interface for generating text
using Markov chains with various configuration options.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from markov_generator import MarkovGenerator, create_sample_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveMarkovGenerator:
    """
    Interactive command-line interface for Markov chain text generation.
    """
    
    def __init__(self):
        """Initialize the interactive interface."""
        self.generator = None
        self.current_chain_type = "word"
        self.current_order = 2
        self.current_settings = {
            "length": 100,
            "temperature": 1.0,
            "max_attempts": 1000,
            "start_with_prompt": True
        }
        self.training_data_files = []
        self.current_data_file = None
        
        # Create sample data if it doesn't exist
        self._ensure_sample_data()
        
        print("📝 Task-03: Text Generation with Markov Chains")
        print("=" * 60)
        print("Type 'help' for available commands")
        print("Type 'quit' to exit")
        print()
    
    def _ensure_sample_data(self):
        """Ensure sample training data exists."""
        data_file = Path("data/sample_texts.txt")
        if not data_file.exists():
            create_sample_data()
            self.training_data_files.append(str(data_file))
            self.current_data_file = str(data_file)
    
    def initialize_generator(self, chain_type: str = "word", order: int = 2):
        """Initialize the Markov chain generator."""
        try:
            if (self.generator is None or 
                chain_type != self.current_chain_type or 
                order != self.current_order):
                
                print(f"🔄 Initializing {chain_type} Markov chain (order={order})...")
                self.generator = MarkovGenerator(
                    chain_type=chain_type,
                    order=order
                )
                self.current_chain_type = chain_type
                self.current_order = order
                print(f"✅ {chain_type} Markov chain initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Error initializing Markov chain: {e}")
            return False
    
    def run(self):
        """Run the interactive interface."""
        while True:
            try:
                command = input("\n📝 MarkovGen> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    print("👋 Goodbye!")
                    break
                elif command == "help":
                    self.show_help()
                elif command == "generate":
                    self.generate_text_interactive()
                elif command == "train":
                    self.train_model_interactive()
                elif command == "settings":
                    self.show_settings()
                elif command == "change_chain":
                    self.change_chain_interactive()
                elif command == "load_data":
                    self.load_data_interactive()
                elif command == "list_data":
                    self.list_data_files()
                elif command == "save_model":
                    self.save_model_interactive()
                elif command == "load_model":
                    self.load_model_interactive()
                elif command == "info":
                    self.show_info()
                elif command == "test":
                    self.run_test()
                elif command.startswith("generate "):
                    # Direct generation with prompt
                    prompt = command[9:]  # Remove "generate " prefix
                    self.generate_with_prompt(prompt)
                else:
                    print("❓ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_help(self):
        """Show help information."""
        help_text = """
📝 Available Commands:

🎯 Generation:
  generate              - Generate text with interactive prompt input
  generate <prompt>     - Generate text with specified prompt

🏋️ Training & Data:
  train                 - Train model on current data file
  load_data             - Load training data from file
  list_data             - Show available training data files

⚙️ Settings & Configuration:
  settings              - Show current generation settings
  change_chain          - Change chain type and order
  info                  - Show model and system information

💾 Model Management:
  save_model            - Save trained model to file
  load_model            - Load model from file

🧪 Testing:
  test                  - Run a quick test generation
  help                  - Show this help message
  quit                  - Exit the program

💡 Tips:
  - Train on diverse, high-quality text for better results
  - Adjust temperature for creativity vs coherence
  - Use word-level chains for more readable output
  - Try different chain orders (1-3 recommended)
        """
        print(help_text)
    
    def generate_text_interactive(self):
        """Generate text with interactive prompt input."""
        if not self._ensure_trained():
            return
        
        print("\n📝 Enter your starting text (or press Enter for random start):")
        prompt = input("Prompt: ").strip()
        
        self.generate_with_prompt(prompt)
    
    def generate_with_prompt(self, prompt: str):
        """Generate text with the given prompt."""
        if not self._ensure_trained():
            return
        
        print(f"\n📝 Generating text with prompt: '{prompt}'")
        print(f"📊 Settings: length={self.current_settings['length']}, "
              f"temperature={self.current_settings['temperature']}")
        
        try:
            # Generate text
            generated_text = self.generator.generate_text(
                prompt=prompt,
                length=self.current_settings["length"],
                temperature=self.current_settings["temperature"],
                max_attempts=self.current_settings["max_attempts"],
                start_with_prompt=self.current_settings["start_with_prompt"]
            )
            
            # Save generated text
            self._save_generated_text(generated_text, prompt)
            
            print(f"✅ Text generated successfully!")
            print(f"📄 Generated text: {generated_text}")
            
        except Exception as e:
            print(f"❌ Error generating text: {e}")
    
    def _save_generated_text(self, text: str, prompt: str):
        """Save generated text to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_texts/markov_generated_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Chain Type: {self.current_chain_type}\n")
            f.write(f"Order: {self.current_order}\n")
            f.write(f"Temperature: {self.current_settings['temperature']}\n")
            f.write("-" * 50 + "\n")
            f.write(text + "\n")
        
        print(f"📁 Saved to: {filename}")
    
    def train_model_interactive(self):
        """Train the model interactively."""
        if not self.initialize_generator(self.current_chain_type, self.current_order):
            return
        
        if not self.current_data_file:
            print("❌ No training data loaded. Use 'load_data' first.")
            return
        
        print(f"\n🏋️ Training {self.current_chain_type} Markov chain (order={self.current_order})...")
        print(f"📁 Training on: {self.current_data_file}")
        
        try:
            self.generator.train_from_file(self.current_data_file)
            print("✅ Training completed successfully!")
            
            # Show training statistics
            stats = self.generator.get_model_info()["training_stats"]
            print(f"📊 Training stats:")
            print(f"  Character vocabulary: {stats['char_vocab_size']}")
            print(f"  Word vocabulary: {stats['word_vocab_size']}")
            print(f"  Character transitions: {stats['char_transitions']}")
            print(f"  Word transitions: {stats['word_transitions']}")
            print(f"  Training time: {stats['training_time']:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Error during training: {e}")
    
    def show_settings(self):
        """Show current generation settings."""
        print("\n⚙️ Current Settings:")
        print(f"  Chain type: {self.current_chain_type}")
        print(f"  Order: {self.current_order}")
        print(f"  Generation length: {self.current_settings['length']}")
        print(f"  Temperature: {self.current_settings['temperature']}")
        print(f"  Max attempts: {self.current_settings['max_attempts']}")
        print(f"  Start with prompt: {self.current_settings['start_with_prompt']}")
        print(f"  Training data: {self.current_data_file or 'None'}")
        
        # Ask if user wants to change settings
        change = input("\nWould you like to change settings? (y/n): ").strip().lower()
        if change == 'y':
            self.change_settings_interactive()
    
    def change_settings_interactive(self):
        """Change settings interactively."""
        print("\n🔧 Change Settings:")
        
        # Generation length
        try:
            length = input(f"Generation length (current: {self.current_settings['length']}): ").strip()
            if length:
                self.current_settings['length'] = int(length)
        except ValueError:
            print("❌ Invalid length. Keeping current setting.")
        
        # Temperature
        try:
            temp = input(f"Temperature (current: {self.current_settings['temperature']}): ").strip()
            if temp:
                self.current_settings['temperature'] = float(temp)
        except ValueError:
            print("❌ Invalid temperature. Keeping current setting.")
        
        # Max attempts
        try:
            attempts = input(f"Max attempts (current: {self.current_settings['max_attempts']}): ").strip()
            if attempts:
                self.current_settings['max_attempts'] = int(attempts)
        except ValueError:
            print("❌ Invalid max attempts. Keeping current setting.")
        
        # Start with prompt
        start_prompt = input(f"Start with prompt (current: {self.current_settings['start_with_prompt']}): ").strip().lower()
        if start_prompt in ['y', 'yes', 'true']:
            self.current_settings['start_with_prompt'] = True
        elif start_prompt in ['n', 'no', 'false']:
            self.current_settings['start_with_prompt'] = False
        
        print("✅ Settings updated!")
    
    def change_chain_interactive(self):
        """Change chain type and order interactively."""
        print("\n🔗 Change Chain Type:")
        print("  1. character - Character-level Markov chain")
        print("  2. word - Word-level Markov chain (recommended)")
        print("  3. hybrid - Hybrid approach")
        
        choice = input("Select chain type (1-3): ").strip()
        
        if choice == "1":
            new_chain_type = "character"
        elif choice == "2":
            new_chain_type = "word"
        elif choice == "3":
            new_chain_type = "hybrid"
        else:
            print("❌ Invalid choice. Keeping current chain type.")
            return
        
        # Chain order
        try:
            order = input(f"Chain order (current: {self.current_order}): ").strip()
            if order:
                new_order = int(order)
            else:
                new_order = self.current_order
        except ValueError:
            print("❌ Invalid order. Keeping current order.")
            new_order = self.current_order
        
        if new_chain_type != self.current_chain_type or new_order != self.current_order:
            self.current_chain_type = new_chain_type
            self.current_order = new_order
            self.generator = None  # Force reinitialization
            print(f"✅ Chain changed to: {new_chain_type} (order={new_order})")
        else:
            print("✅ Already using selected chain type and order.")
    
    def load_data_interactive(self):
        """Load training data interactively."""
        print("\n📁 Load Training Data:")
        
        # Show available data files
        data_dir = Path("data")
        if data_dir.exists():
            data_files = list(data_dir.glob("*.txt"))
            if data_files:
                print("Available data files:")
                for i, file in enumerate(data_files):
                    marker = " → " if str(file) == self.current_data_file else "   "
                    print(f"{marker}{i+1}. {file.name}")
            else:
                print("No data files found in data/ directory.")
        
        filename = input("Enter filename (or press Enter for sample_texts.txt): ").strip()
        if not filename:
            filename = "sample_texts.txt"
        
        filepath = data_dir / filename
        if filepath.exists():
            self.current_data_file = str(filepath)
            if str(filepath) not in self.training_data_files:
                self.training_data_files.append(str(filepath))
            print(f"✅ Loaded: {filename}")
        else:
            print(f"❌ File not found: {filename}")
    
    def list_data_files(self):
        """List available training data files."""
        print("\n📁 Available Training Data Files:")
        
        data_dir = Path("data")
        if data_dir.exists():
            data_files = list(data_dir.glob("*.txt"))
            if data_files:
                for i, file in enumerate(data_files):
                    marker = " → " if str(file) == self.current_data_file else "   "
                    size = file.stat().st_size
                    print(f"{marker}{i+1}. {file.name} ({size} bytes)")
            else:
                print("No data files found.")
        else:
            print("Data directory not found.")
    
    def save_model_interactive(self):
        """Save the trained model interactively."""
        if not self.generator or not self.generator._is_trained():
            print("❌ No trained model to save. Train a model first.")
            return
        
        filename = input("Enter model filename (or press Enter for auto-generated): ").strip()
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"markov_model_{self.current_chain_type}_{self.current_order}_{timestamp}.json"
        
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = f"models/{filename}"
        try:
            self.generator.save_model(filepath)
            print(f"✅ Model saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def load_model_interactive(self):
        """Load a trained model interactively."""
        print("\n📁 Load Model:")
        
        # Show available model files
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.json"))
            if model_files:
                print("Available model files:")
                for i, file in enumerate(model_files):
                    print(f"  {i+1}. {file.name}")
            else:
                print("No model files found in models/ directory.")
        
        filename = input("Enter model filename: ").strip()
        if not filename.endswith('.json'):
            filename += '.json'
        
        filepath = models_dir / filename
        if filepath.exists():
            try:
                self.generator = MarkovGenerator(model_path=str(filepath))
                info = self.generator.get_model_info()
                self.current_chain_type = info["chain_type"]
                self.current_order = info["order"]
                print(f"✅ Model loaded: {filename}")
                print(f"📊 Chain type: {info['chain_type']}, Order: {info['order']}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
        else:
            print(f"❌ File not found: {filename}")
    
    def show_info(self):
        """Show model and system information."""
        print("\nℹ️ System Information:")
        
        if self.generator:
            info = self.generator.get_model_info()
            print(f"  Chain type: {info['chain_type']}")
            print(f"  Order: {info['order']}")
            print(f"  Smoothing: {info['smoothing']}")
            print(f"  Is trained: {info['is_trained']}")
            
            if info['is_trained']:
                stats = info['training_stats']
                print(f"  Character vocabulary: {stats['char_vocab_size']}")
                print(f"  Word vocabulary: {stats['word_vocab_size']}")
                print(f"  Character transitions: {stats['char_transitions']}")
                print(f"  Word transitions: {stats['word_transitions']}")
        else:
            print("  Model: Not initialized")
        
        print(f"  Current chain type: {self.current_chain_type}")
        print(f"  Current order: {self.current_order}")
        print(f"  Training data: {self.current_data_file or 'None'}")
        print(f"  Available data files: {len(self.training_data_files)}")
    
    def run_test(self):
        """Run a quick test generation."""
        print("\n🧪 Running Test Generation...")
        
        if not self._ensure_trained():
            return
        
        test_prompt = "The future of"
        print(f"Test prompt: '{test_prompt}'")
        
        try:
            # Use minimal settings for quick test
            generated_text = self.generator.generate_text(
                prompt=test_prompt,
                length=30,
                temperature=1.0
            )
            
            print(f"✅ Test successful!")
            print(f"📄 Generated: {generated_text}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    def _ensure_trained(self) -> bool:
        """Ensure the model is trained before generation."""
        if not self.generator:
            if not self.initialize_generator(self.current_chain_type, self.current_order):
                return False
        
        if not self.generator._is_trained():
            print("❌ Model not trained. Use 'train' command first.")
            return False
        
        return True


def main():
    """Main function to run the interactive interface."""
    parser = argparse.ArgumentParser(description="Interactive Markov Chain Text Generator")
    parser.add_argument("command", nargs="?", help="Command to run")
    parser.add_argument("--chain-type", default="word", help="Chain type to use")
    parser.add_argument("--order", type=int, default=2, help="Chain order")
    parser.add_argument("--prompt", help="Text prompt for generation")
    
    args = parser.parse_args()
    
    if args.command == "test":
        # Run test mode
        generator = InteractiveMarkovGenerator()
        generator.run_test()
    elif args.prompt:
        # Generate text with provided prompt
        generator = InteractiveMarkovGenerator()
        if generator.initialize_generator(args.chain_type, args.order):
            generator.generate_with_prompt(args.prompt)
    else:
        # Run interactive mode
        generator = InteractiveMarkovGenerator()
        generator.run()


if __name__ == "__main__":
    main()











