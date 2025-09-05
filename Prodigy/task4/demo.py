"""
Task-04 Demo: Image-to-Image Translation with cGAN
Demonstration script showing the capabilities of the pix2pix image translation system.

This demo showcases the interface and functionality with a simplified implementation
that demonstrates the core concepts without requiring heavy dependencies.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import random
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class DemoPix2PixTranslator:
    """
    Demo version of the Pix2PixTranslator that creates sample translated images.
    This demonstrates the interface without requiring heavy dependencies.
    """
    
    def __init__(self, translation_type: str = "sketch_to_photo", image_size: tuple = (256, 256)):
        self.translation_type = translation_type
        self.image_size = image_size
        self.is_trained = False
        
        # Create output directories
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("generated_images").mkdir(exist_ok=True)
        Path("sample_data").mkdir(exist_ok=True)
        
        print(f"🎨 Demo Pix2PixTranslator initialized for {translation_type} (size={image_size})")
    
    def train(self, data_dir: str, num_epochs: int = 50, batch_size: int = 1, **kwargs):
        """Simulate training on data."""
        print(f"🏋️ Training pix2pix model for {self.translation_type}...")
        print(f"📁 Training on: {data_dir}")
        print(f"📊 Parameters: epochs={num_epochs}, batch_size={batch_size}")
        
        # Simulate training progress
        for epoch in range(min(num_epochs, 5)):  # Show first 5 epochs
            time.sleep(0.5)
            g_loss = random.uniform(0.5, 2.0)
            d_loss = random.uniform(0.3, 1.5)
            print(f"  Epoch [{epoch+1}/{num_epochs}] - G Loss: {g_loss:.4f}, D Loss: {d_loss:.4f}")
        
        self.is_trained = True
        print("✅ Training completed!")
    
    def translate_image(self, input_path: str, output_path: str = None, translation_type: str = None, **kwargs) -> str:
        """Generate demo translated image based on input."""
        print(f"🎨 Translating image: {input_path}")
        
        if translation_type:
            self.translation_type = translation_type
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"generated_images/{self.translation_type}_{timestamp}.jpg"
        
        # Create demo translated image
        translated_image = self._create_demo_translation(input_path)
        
        # Save the image
        translated_image.save(output_path)
        print(f"✅ Image translated: {output_path}")
        
        return output_path
    
    def _create_demo_translation(self, input_path: str) -> Image.Image:
        """Create a demo translated image based on the translation type."""
        # Create a base image
        img = Image.new('RGB', self.image_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Add text based on translation type
        text = f"Demo {self.translation_type.replace('_', ' ').title()}"
        
        # Try to use a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Calculate text position (center)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (self.image_size[0] - text_width) // 2
        y = (self.image_size[1] - text_height) // 2
        
        # Add background color based on translation type
        colors = {
            "sketch_to_photo": "lightblue",
            "day_to_night": "darkblue",
            "bw_to_color": "pink",
            "style_transfer": "lightgreen"
        }
        
        bg_color = colors.get(self.translation_type, "lightgray")
        img = Image.new('RGB', self.image_size, color=bg_color)
        draw = ImageDraw.Draw(img)
        
        # Draw text
        draw.text((x, y), text, fill='black', font=font)
        
        # Add some decorative elements based on translation type
        if self.translation_type == "sketch_to_photo":
            # Add some simple shapes to simulate a photo
            draw.rectangle([50, 50, 100, 100], fill='red', outline='black')
            draw.ellipse([150, 50, 200, 100], fill='blue', outline='black')
        elif self.translation_type == "day_to_night":
            # Add stars for night effect
            for _ in range(20):
                x_star = random.randint(0, self.image_size[0])
                y_star = random.randint(0, self.image_size[1] // 2)
                draw.ellipse([x_star, y_star, x_star+2, y_star+2], fill='yellow')
        elif self.translation_type == "bw_to_color":
            # Add colorful elements
            colors_list = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
            for i in range(5):
                x = random.randint(0, self.image_size[0] - 30)
                y = random.randint(0, self.image_size[1] - 30)
                color = random.choice(colors_list)
                draw.rectangle([x, y, x+30, y+30], fill=color)
        
        return img
    
    def translate_batch(self, input_dir: str, output_dir: str = None, translation_type: str = None, **kwargs) -> list:
        """Translate multiple images in batch."""
        if output_dir is None:
            output_dir = "generated_images"
        
        Path(output_dir).mkdir(exist_ok=True)
        
        input_paths = list(Path(input_dir).glob("*.jpg")) + list(Path(input_dir).glob("*.png"))
        output_paths = []
        
        print(f"🎨 Translating {len(input_paths)} images...")
        
        for input_path in input_paths:
            output_filename = f"{self.translation_type}_{input_path.stem}.jpg"
            output_path = Path(output_dir) / output_filename
            
            try:
                translated_path = self.translate_image(
                    str(input_path),
                    str(output_path),
                    translation_type
                )
                output_paths.append(translated_path)
            except Exception as e:
                print(f"❌ Error translating {input_path}: {e}")
        
        print(f"✅ Batch translation completed: {len(output_paths)} images")
        return output_paths
    
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
            "translation_type": self.translation_type,
            "image_size": self.image_size,
            "generator_type": "unet",
            "discriminator_type": "patch",
            "lambda_l1": 100.0,
            "lambda_gan": 1.0,
            "device": "cpu",
            "training_stats": {
                "epochs_trained": 50 if self.is_trained else 0,
                "total_loss": 0.0,
                "g_loss": 1.2 if self.is_trained else 0.0,
                "d_loss": 0.8 if self.is_trained else 0.0,
                "training_time": 120.5 if self.is_trained else 0.0
            }
        }

def create_sample_data():
    """Create sample training data."""
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample images
    sample_images = [
        ("sketch_input.jpg", "sketch_output.jpg"),
        ("day_input.jpg", "night_output.jpg"),
        ("bw_input.jpg", "color_output.jpg"),
        ("style_input.jpg", "style_output.jpg")
    ]
    
    for input_name, output_name in sample_images:
        # Create simple test images
        input_img = Image.new('RGB', (256, 256), color='white')
        output_img = Image.new('RGB', (256, 256), color='lightblue')
        
        # Add some content to input images
        draw_input = ImageDraw.Draw(input_img)
        draw_input.rectangle([50, 50, 200, 200], outline='black', width=3)
        draw_input.text((100, 120), "Input", fill='black')
        
        # Add some content to output images
        draw_output = ImageDraw.Draw(output_img)
        draw_output.rectangle([50, 50, 200, 200], fill='lightgreen', outline='black', width=2)
        draw_output.text((100, 120), "Output", fill='black')
        
        input_img.save(sample_dir / input_name)
        output_img.save(sample_dir / output_name)
    
    print(f"✅ Sample data created in {sample_dir}")

def run_interactive_demo():
    """Run an interactive demo of the pix2pix system."""
    print("🎨 Task-04 Demo: Image-to-Image Translation with cGAN")
    print("=" * 60)
    print("This is a demonstration of the pix2pix image translation interface.")
    print("Images generated are simulated - the real system uses actual cGAN models.")
    print()
    
    translator = DemoPix2PixTranslator()
    
    # Sample translation types
    translation_types = [
        "sketch_to_photo",
        "day_to_night", 
        "bw_to_color",
        "style_transfer"
    ]
    
    while True:
        print("\n🎨 Demo Commands:")
        print("  1. train - Train the model on sample data")
        print("  2. translate <file> - Translate image with file path")
        print("  3. batch - Translate multiple sample images")
        print("  4. info - Show system information")
        print("  5. quit - Exit demo")
        
        command = input("\n🎨 Demo> ").strip().lower()
        
        if command == "quit" or command == "exit":
            print("👋 Demo completed!")
            break
        elif command == "info":
            info = translator.get_model_info()
            print(f"\nℹ️ System Information:")
            print(f"  Translation type: {info['translation_type']}")
            print(f"  Image size: {info['image_size']}")
            print(f"  Generator type: {info['generator_type']}")
            print(f"  Discriminator type: {info['discriminator_type']}")
            print(f"  Is trained: {translator.is_trained}")
            print(f"  Status: Demo mode - simulated image translation")
        elif command == "train":
            print(f"\n🏋️ Training demo model...")
            translator.train("sample_data", num_epochs=10, batch_size=1)
        elif command == "batch":
            print(f"\n🎨 Translating sample images...")
            translator.translate_batch("sample_data", translation_type="sketch_to_photo")
        elif command.startswith("translate "):
            file_path = command[10:]  # Remove "translate " prefix
            if file_path:
                if os.path.exists(file_path):
                    translator.translate_image(file_path)
                else:
                    print(f"❌ File not found: {file_path}")
            else:
                print("❌ Please provide a file path after 'translate'")
        else:
            print("❓ Unknown command. Type 'quit' to exit.")

def run_batch_demo():
    """Run a batch demo showing multiple translations."""
    print("🎨 Task-04 Batch Demo")
    print("=" * 40)
    
    translator = DemoPix2PixTranslator()
    
    # Train the model
    print("🏋️ Training demo model...")
    translator.train("sample_data", num_epochs=5, batch_size=1)
    
    # Sample translation types
    translation_types = [
        "sketch_to_photo",
        "day_to_night",
        "bw_to_color",
        "style_transfer"
    ]
    
    print(f"🎨 Processing {len(translation_types)} translation types...")
    
    for i, trans_type in enumerate(translation_types):
        print(f"\n🎨 [{i+1}/{len(translation_types)}] Translating: {trans_type}")
        
        # Set translation type
        translator.translation_type = trans_type
        
        # Translate sample image
        if os.path.exists("sample_data/sketch_input.jpg"):
            output_path = translator.translate_image(
                "sample_data/sketch_input.jpg",
                f"generated_images/demo_batch_{i+1:02d}_{trans_type}.jpg",
                translation_type=trans_type
            )
            print(f"✅ Completed: {output_path}")
    
    print(f"\n🎉 Batch demo completed! Generated {len(translation_types)} translated images.")

def show_system_info():
    """Show system information and capabilities."""
    print("🎨 Task-04: Pix2Pix Image Translation System Information")
    print("=" * 50)
    
    print("\n📋 System Capabilities:")
    print("  ✅ Image-to-image translation using cGAN")
    print("  ✅ Pix2Pix architecture implementation")
    print("  ✅ Multiple translation types (sketch-to-photo, day-to-night, etc.)")
    print("  ✅ Interactive command-line interface")
    print("  ✅ Batch image processing")
    print("  ✅ Model training and management")
    print("  ✅ Real-time image translation")
    
    print("\n🎯 Supported Translation Types:")
    print("  🎨 Sketch-to-Photo - Convert sketches to realistic photos")
    print("  🌙 Day-to-Night - Transform daytime scenes to nighttime")
    print("  🌈 B&W-to-Color - Colorize black and white images")
    print("  🎭 Style-Transfer - Apply artistic styles to photos")
    
    print("\n⚙️ Model Architecture:")
    print("  🧠 Generator: U-Net architecture with skip connections")
    print("  🔍 Discriminator: PatchGAN for local patch discrimination")
    print("  📊 Loss Functions: GAN loss + L1 reconstruction loss")
    print("  🎯 Optimization: Adam optimizer with specific hyperparameters")
    
    print("\n📁 File Structure:")
    print("  📂 data/ - Training data directories")
    print("  📂 models/ - Saved cGAN models")
    print("  📂 generated_images/ - Output directory")
    print("  📂 sample_data/ - Sample image pairs")
    
    print("\n🚀 Quick Start:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run interactive: python interactive_translator.py")
    print("  3. Translate image: python interactive_translator.py --input 'image.jpg'")
    print("  4. Run test: python interactive_translator.py test")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Task-04 Demo: Pix2Pix Image Translation")
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
        print("🎨 Task-04 Demo: Image-to-Image Translation with cGAN")
        print("=" * 60)
        print("Available modes:")
        print("  interactive - Run interactive demo")
        print("  batch - Run batch translation demo")
        print("  info - Show system information")
        print("\nExample: python demo.py interactive")











