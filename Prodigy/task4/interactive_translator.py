"""
Interactive Pix2Pix Image Translator Interface
Task-04: Command-line interface for image-to-image translation

This module provides an interactive command-line interface for translating images
using the pix2pix architecture with various configuration options.
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

from pix2pix_translator import Pix2PixTranslator, create_sample_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractivePix2PixTranslator:
    """
    Interactive command-line interface for pix2pix image translation.
    """
    
    def __init__(self):
        """Initialize the interactive interface."""
        self.translator = None
        self.current_translation_type = "sketch_to_photo"
        self.current_settings = {
            "image_size": (256, 256),
            "generator_type": "unet",
            "discriminator_type": "patch",
            "lambda_l1": 100.0,
            "lambda_gan": 1.0,
            "learning_rate": 0.0002,
            "beta1": 0.5
        }
        self.sample_data_files = []
        self.current_data_dir = None
        
        # Create sample data if it doesn't exist
        self._ensure_sample_data()
        
        print("🎨 Task-04: Image-to-Image Translation with cGAN")
        print("=" * 60)
        print("Type 'help' for available commands")
        print("Type 'quit' to exit")
        print()
    
    def _ensure_sample_data(self):
        """Ensure sample training data exists."""
        sample_dir = Path("sample_data")
        if not sample_dir.exists() or not list(sample_dir.glob("*.jpg")):
            create_sample_data()
            self.sample_data_files = [str(f) for f in sample_dir.glob("*.jpg")]
            self.current_data_dir = str(sample_dir)
    
    def initialize_translator(self, translation_type: str = "sketch_to_photo"):
        """Initialize the pix2pix translator."""
        try:
            if (self.translator is None or 
                translation_type != self.current_translation_type):
                
                print(f"🔄 Initializing pix2pix translator for {translation_type}...")
                self.translator = Pix2PixTranslator(
                    translation_type=translation_type,
                    image_size=self.current_settings["image_size"],
                    generator_type=self.current_settings["generator_type"],
                    discriminator_type=self.current_settings["discriminator_type"],
                    lambda_l1=self.current_settings["lambda_l1"],
                    lambda_gan=self.current_settings["lambda_gan"],
                    learning_rate=self.current_settings["learning_rate"],
                    beta1=self.current_settings["beta1"]
                )
                self.current_translation_type = translation_type
                print(f"✅ Pix2Pix translator initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Error initializing translator: {e}")
            return False
    
    def run(self):
        """Run the interactive interface."""
        while True:
            try:
                command = input("\n🎨 Pix2Pix> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    print("👋 Goodbye!")
                    break
                elif command == "help":
                    self.show_help()
                elif command == "translate":
                    self.translate_image_interactive()
                elif command == "train":
                    self.train_model_interactive()
                elif command == "settings":
                    self.show_settings()
                elif command == "change_type":
                    self.change_translation_type_interactive()
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
                elif command.startswith("translate "):
                    # Direct translation with file path
                    file_path = command[10:]  # Remove "translate " prefix
                    self.translate_with_path(file_path)
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
🎨 Available Commands:

🎯 Translation:
  translate              - Translate image with interactive file input
  translate <file_path>  - Translate image with specified file path

🏋️ Training & Data:
  train                 - Train model on current data directory
  load_data             - Load training data from directory
  list_data             - Show available training data files

⚙️ Settings & Configuration:
  settings              - Show current translation settings
  change_type           - Change translation type
  info                  - Show model and system information

💾 Model Management:
  save_model            - Save trained model to file
  load_model            - Load model from file

🧪 Testing:
  test                  - Run a quick test translation
  help                  - Show this help message
  quit                  - Exit the program

💡 Tips:
  - Use high-quality paired images for better training
  - Adjust lambda_l1 and lambda_gan for different translation tasks
  - Use GPU acceleration for faster training and inference
  - Try different image sizes based on your needs
        """
        print(help_text)
    
    def translate_image_interactive(self):
        """Translate image with interactive file input."""
        if not self._ensure_initialized():
            return
        
        print("\n🎨 Enter the path to your input image:")
        input_path = input("Input image path: ").strip()
        
        if not input_path:
            print("❌ No input path provided.")
            return
        
        self.translate_with_path(input_path)
    
    def translate_with_path(self, input_path: str):
        """Translate image with the given path."""
        if not self._ensure_initialized():
            return
        
        if not os.path.exists(input_path):
            print(f"❌ File not found: {input_path}")
            return
        
        print(f"\n🎨 Translating image: {input_path}")
        print(f"📊 Settings: type={self.current_translation_type}, size={self.current_settings['image_size']}")
        
        try:
            # Generate output path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"generated_images/{self.current_translation_type}_{timestamp}.jpg"
            
            # Translate image
            translated_path = self.translator.translate_image(
                input_path=input_path,
                output_path=output_path,
                translation_type=self.current_translation_type
            )
            
            print(f"✅ Image translated successfully!")
            print(f"📁 Saved to: {translated_path}")
            
        except Exception as e:
            print(f"❌ Error translating image: {e}")
    
    def train_model_interactive(self):
        """Train the model interactively."""
        if not self.initialize_translator(self.current_translation_type):
            return
        
        if not self.current_data_dir:
            print("❌ No training data loaded. Use 'load_data' first.")
            return
        
        print(f"\n🏋️ Training pix2pix model for {self.current_translation_type}...")
        print(f"📁 Training on: {self.current_data_dir}")
        
        # Get training parameters
        try:
            num_epochs = input("Number of epochs (default: 50): ").strip()
            num_epochs = int(num_epochs) if num_epochs else 50
            
            batch_size = input("Batch size (default: 1): ").strip()
            batch_size = int(batch_size) if batch_size else 1
            
        except ValueError:
            print("❌ Invalid input. Using default values.")
            num_epochs = 50
            batch_size = 1
        
        try:
            self.translator.train(
                data_dir=self.current_data_dir,
                num_epochs=num_epochs,
                batch_size=batch_size
            )
            print("✅ Training completed successfully!")
            
            # Show training statistics
            stats = self.translator.get_model_info()["training_stats"]
            print(f"📊 Training stats:")
            print(f"  Epochs trained: {stats['epochs_trained']}")
            print(f"  Generator loss: {stats['g_loss']:.4f}")
            print(f"  Discriminator loss: {stats['d_loss']:.4f}")
            print(f"  Training time: {stats['training_time']:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Error during training: {e}")
    
    def show_settings(self):
        """Show current translation settings."""
        print("\n⚙️ Current Settings:")
        print(f"  Translation type: {self.current_translation_type}")
        print(f"  Image size: {self.current_settings['image_size']}")
        print(f"  Generator type: {self.current_settings['generator_type']}")
        print(f"  Discriminator type: {self.current_settings['discriminator_type']}")
        print(f"  Lambda L1: {self.current_settings['lambda_l1']}")
        print(f"  Lambda GAN: {self.current_settings['lambda_gan']}")
        print(f"  Learning rate: {self.current_settings['learning_rate']}")
        print(f"  Beta1: {self.current_settings['beta1']}")
        print(f"  Training data: {self.current_data_dir or 'None'}")
        
        # Ask if user wants to change settings
        change = input("\nWould you like to change settings? (y/n): ").strip().lower()
        if change == 'y':
            self.change_settings_interactive()
    
    def change_settings_interactive(self):
        """Change settings interactively."""
        print("\n🔧 Change Settings:")
        
        # Image size
        try:
            size_input = input(f"Image size (current: {self.current_settings['image_size']}): ").strip()
            if size_input:
                width, height = map(int, size_input.split('x'))
                self.current_settings['image_size'] = (width, height)
        except ValueError:
            print("❌ Invalid size format. Use WxH (e.g., 256x256). Keeping current setting.")
        
        # Lambda L1
        try:
            lambda_l1 = input(f"Lambda L1 (current: {self.current_settings['lambda_l1']}): ").strip()
            if lambda_l1:
                self.current_settings['lambda_l1'] = float(lambda_l1)
        except ValueError:
            print("❌ Invalid lambda L1. Keeping current setting.")
        
        # Lambda GAN
        try:
            lambda_gan = input(f"Lambda GAN (current: {self.current_settings['lambda_gan']}): ").strip()
            if lambda_gan:
                self.current_settings['lambda_gan'] = float(lambda_gan)
        except ValueError:
            print("❌ Invalid lambda GAN. Keeping current setting.")
        
        # Learning rate
        try:
            lr = input(f"Learning rate (current: {self.current_settings['learning_rate']}): ").strip()
            if lr:
                self.current_settings['learning_rate'] = float(lr)
        except ValueError:
            print("❌ Invalid learning rate. Keeping current setting.")
        
        print("✅ Settings updated!")
    
    def change_translation_type_interactive(self):
        """Change translation type interactively."""
        print("\n🔄 Change Translation Type:")
        print("  1. sketch_to_photo - Convert sketches to photos")
        print("  2. day_to_night - Transform day to night")
        print("  3. bw_to_color - Colorize black and white images")
        print("  4. style_transfer - Apply artistic styles")
        print("  5. custom - Custom translation task")
        
        choice = input("Select translation type (1-5): ").strip()
        
        translation_types = {
            "1": "sketch_to_photo",
            "2": "day_to_night",
            "3": "bw_to_color",
            "4": "style_transfer",
            "5": "custom"
        }
        
        if choice in translation_types:
            new_type = translation_types[choice]
            if new_type == "custom":
                new_type = input("Enter custom translation type: ").strip()
            
            if new_type != self.current_translation_type:
                self.current_translation_type = new_type
                self.translator = None  # Force reinitialization
                print(f"✅ Translation type changed to: {new_type}")
            else:
                print("✅ Already using selected translation type.")
        else:
            print("❌ Invalid choice. Keeping current translation type.")
    
    def load_data_interactive(self):
        """Load training data interactively."""
        print("\n📁 Load Training Data:")
        
        # Show available data directories
        data_dirs = [d for d in Path(".").iterdir() if d.is_dir() and d.name in ["data", "sample_data"]]
        if data_dirs:
            print("Available data directories:")
            for i, data_dir in enumerate(data_dirs):
                marker = " → " if str(data_dir) == self.current_data_dir else "   "
                print(f"{marker}{i+1}. {data_dir.name}")
        else:
            print("No data directories found.")
        
        data_path = input("Enter data directory path (or press Enter for sample_data): ").strip()
        if not data_path:
            data_path = "sample_data"
        
        data_dir = Path(data_path)
        if data_dir.exists() and data_dir.is_dir():
            self.current_data_dir = str(data_dir)
            if str(data_dir) not in self.sample_data_files:
                self.sample_data_files.append(str(data_dir))
            print(f"✅ Loaded: {data_path}")
        else:
            print(f"❌ Directory not found: {data_path}")
    
    def list_data_files(self):
        """List available training data files."""
        print("\n📁 Available Training Data Files:")
        
        if self.current_data_dir:
            data_dir = Path(self.current_data_dir)
            if data_dir.exists():
                image_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
                if image_files:
                    for i, file in enumerate(image_files):
                        marker = " → " if str(file) in self.sample_data_files else "   "
                        size = file.stat().st_size
                        print(f"{marker}{i+1}. {file.name} ({size} bytes)")
                else:
                    print("No image files found.")
            else:
                print("Data directory not found.")
        else:
            print("No data directory loaded.")
    
    def save_model_interactive(self):
        """Save the trained model interactively."""
        if not self.translator:
            print("❌ No translator initialized. Initialize a translator first.")
            return
        
        filename = input("Enter model filename (or press Enter for auto-generated): ").strip()
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pix2pix_{self.current_translation_type}_{timestamp}.pth"
        
        if not filename.endswith('.pth'):
            filename += '.pth'
        
        filepath = f"models/{filename}"
        try:
            self.translator.save_model(filepath)
            print(f"✅ Model saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    def load_model_interactive(self):
        """Load a trained model interactively."""
        print("\n📁 Load Model:")
        
        # Show available model files
        models_dir = Path("models")
        if models_dir.exists():
            model_files = list(models_dir.glob("*.pth"))
            if model_files:
                print("Available model files:")
                for i, file in enumerate(model_files):
                    print(f"  {i+1}. {file.name}")
            else:
                print("No model files found in models/ directory.")
        
        filename = input("Enter model filename: ").strip()
        if not filename.endswith('.pth'):
            filename += '.pth'
        
        filepath = models_dir / filename
        if filepath.exists():
            try:
                if not self.translator:
                    self.initialize_translator()
                
                self.translator.load_model(str(filepath))
                info = self.translator.get_model_info()
                self.current_translation_type = info["translation_type"]
                print(f"✅ Model loaded: {filename}")
                print(f"📊 Translation type: {info['translation_type']}, Image size: {info['image_size']}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
        else:
            print(f"❌ File not found: {filename}")
    
    def show_info(self):
        """Show model and system information."""
        print("\nℹ️ System Information:")
        
        if self.translator:
            info = self.translator.get_model_info()
            print(f"  Translation type: {info['translation_type']}")
            print(f"  Image size: {info['image_size']}")
            print(f"  Generator type: {info['generator_type']}")
            print(f"  Discriminator type: {info['discriminator_type']}")
            print(f"  Lambda L1: {info['lambda_l1']}")
            print(f"  Lambda GAN: {info['lambda_gan']}")
            print(f"  Device: {info['device']}")
            
            if info['training_stats']['epochs_trained'] > 0:
                stats = info['training_stats']
                print(f"  Epochs trained: {stats['epochs_trained']}")
                print(f"  Generator loss: {stats['g_loss']:.4f}")
                print(f"  Discriminator loss: {stats['d_loss']:.4f}")
        else:
            print("  Translator: Not initialized")
        
        print(f"  Current translation type: {self.current_translation_type}")
        print(f"  Training data: {self.current_data_dir or 'None'}")
        print(f"  Available data files: {len(self.sample_data_files)}")
    
    def run_test(self):
        """Run a quick test translation."""
        print("\n🧪 Running Test Translation...")
        
        if not self._ensure_initialized():
            return
        
        # Check if sample data exists
        test_input = "sample_data/sketch_input.jpg"
        if not os.path.exists(test_input):
            print(f"❌ Test file not found: {test_input}")
            return
        
        print(f"Test input: {test_input}")
        
        try:
            # Use minimal settings for quick test
            output_path = self.translator.translate_image(
                input_path=test_input,
                output_path="generated_images/test_translation.jpg",
                translation_type=self.current_translation_type
            )
            
            print(f"✅ Test successful!")
            print(f"📁 Generated: {output_path}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    def _ensure_initialized(self) -> bool:
        """Ensure the translator is initialized before use."""
        if not self.translator:
            if not self.initialize_translator(self.current_translation_type):
                return False
        
        return True


def main():
    """Main function to run the interactive interface."""
    parser = argparse.ArgumentParser(description="Interactive Pix2Pix Image Translator")
    parser.add_argument("command", nargs="?", help="Command to run")
    parser.add_argument("--translation-type", default="sketch_to_photo", help="Translation type to use")
    parser.add_argument("--input", help="Input image path for translation")
    
    args = parser.parse_args()
    
    if args.command == "test":
        # Run test mode
        translator = InteractivePix2PixTranslator()
        translator.run_test()
    elif args.input:
        # Translate image with provided path
        translator = InteractivePix2PixTranslator()
        if translator.initialize_translator(args.translation_type):
            translator.translate_with_path(args.input)
    else:
        # Run interactive mode
        translator = InteractivePix2PixTranslator()
        translator.run()


if __name__ == "__main__":
    main()











