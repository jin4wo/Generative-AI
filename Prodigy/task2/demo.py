"""
Task-02 Demo: Image Generation with Pre-trained Models
Demonstration script showing the capabilities of the image generation system.

This demo showcases the interface and functionality without requiring
heavy model dependencies to be installed.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_demo_image(prompt: str, size: tuple = (512, 512)) -> Image.Image:
    """
    Create a demo image that represents what would be generated.
    This is a placeholder for the actual model generation.
    """
    # Create a gradient background
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Create a simple gradient
    for y in range(size[1]):
        r = int(255 * (1 - y / size[1]))
        g = int(128 * (y / size[1]))
        b = int(255 * (y / size[1]))
        draw.line([(0, y), (size[0], y)], fill=(r, g, b))
    
    # Add some geometric shapes to represent the prompt
    if "sunset" in prompt.lower():
        # Draw a sun
        draw.ellipse([size[0]//2-50, size[1]//2-50, size[0]//2+50, size[1]//2+50], fill='yellow')
    elif "mountain" in prompt.lower():
        # Draw mountains
        points = [(0, size[1]), (size[0]//4, size[1]//2), (size[0]//2, size[1]//3), 
                 (3*size[0]//4, size[1]//2), (size[0], size[1])]
        draw.polygon(points, fill='gray')
    elif "forest" in prompt.lower():
        # Draw trees
        for i in range(5):
            x = (i + 1) * size[0] // 6
            draw.ellipse([x-20, size[1]//2-40, x+20, size[1]//2], fill='green')
            draw.rectangle([x-5, size[1]//2, x+5, size[1]], fill='brown')
    else:
        # Generic shapes
        draw.rectangle([size[0]//4, size[1]//4, 3*size[0]//4, 3*size[1]//4], fill='blue', outline='black')
    
    # Add text label
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    # Add prompt text at the bottom
    text = f"Generated: {prompt[:30]}..."
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (size[0] - text_width) // 2
    draw.text((text_x, size[1] - 30), text, fill='white', font=font)
    
    return img

class DemoImageGenerator:
    """
    Demo version of the ImageGenerator that creates placeholder images.
    This demonstrates the interface without requiring heavy dependencies.
    """
    
    def __init__(self, model_type: str = "demo", image_size: tuple = (512, 512)):
        self.model_type = model_type
        self.image_size = image_size
        self.device = "cpu"
        
        # Create output directories
        Path("generated_images").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("prompts").mkdir(exist_ok=True)
        
        print(f"🎨 Demo ImageGenerator initialized with {model_type} model")
    
    def generate_image(self, prompt: str, num_images: int = 1, **kwargs) -> list:
        """Generate demo images from prompt."""
        print(f"🎨 Generating {num_images} image(s) with prompt: '{prompt}'")
        
        images = []
        for i in range(num_images):
            # Simulate generation time
            time.sleep(1)
            
            # Create demo image
            img = create_demo_image(prompt, self.image_size)
            images.append(img)
            
            print(f"✅ Generated image {i+1}/{num_images}")
        
        return images[0] if num_images == 1 else images
    
    def save_image(self, images, filename: str = None) -> list:
        """Save generated images."""
        if not isinstance(images, list):
            images = [images]
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"demo_generated_{timestamp}"
        
        saved_paths = []
        for i, img in enumerate(images):
            if len(images) > 1:
                filepath = f"generated_images/{filename}_{i+1}.png"
            else:
                filepath = f"generated_images/{filename}.png"
            
            img.save(filepath)
            saved_paths.append(filepath)
            print(f"📁 Saved to: {filepath}")
        
        return saved_paths
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_type": self.model_type,
            "device": self.device,
            "image_size": self.image_size,
            "status": "Demo mode - placeholder images"
        }

def run_interactive_demo():
    """Run an interactive demo of the image generation system."""
    print("🎨 Task-02 Demo: Image Generation with Pre-trained Models")
    print("=" * 60)
    print("This is a demonstration of the image generation interface.")
    print("Images generated are placeholders - install dependencies for real generation.")
    print()
    
    generator = DemoImageGenerator()
    
    # Sample prompts
    sample_prompts = [
        "a beautiful sunset over mountains",
        "a futuristic city with flying cars",
        "a cozy coffee shop interior",
        "a majestic dragon flying over a castle",
        "a serene forest with sunlight"
    ]
    
    while True:
        print("\n🎨 Demo Commands:")
        print("  1. generate <prompt> - Generate image with prompt")
        print("  2. sample - Generate images from sample prompts")
        print("  3. info - Show system information")
        print("  4. quit - Exit demo")
        
        command = input("\n🎨 Demo> ").strip().lower()
        
        if command == "quit" or command == "exit":
            print("👋 Demo completed!")
            break
        elif command == "info":
            info = generator.get_model_info()
            print(f"\nℹ️ System Information:")
            print(f"  Model: {info['model_type']}")
            print(f"  Device: {info['device']}")
            print(f"  Image size: {info['image_size']}")
            print(f"  Status: {info['status']}")
        elif command == "sample":
            print(f"\n🎨 Generating sample images...")
            for i, prompt in enumerate(sample_prompts):
                print(f"\n📝 Sample {i+1}: '{prompt}'")
                images = generator.generate_image(prompt, num_images=1)
                saved_paths = generator.save_image(images, f"sample_{i+1}")
        elif command.startswith("generate "):
            prompt = command[9:]  # Remove "generate " prefix
            if prompt:
                images = generator.generate_image(prompt, num_images=1)
                saved_paths = generator.save_image(images, "demo_generated")
            else:
                print("❌ Please provide a prompt after 'generate'")
        else:
            print("❓ Unknown command. Type 'quit' to exit.")

def run_batch_demo():
    """Run a batch demo showing multiple generations."""
    print("🎨 Task-02 Batch Demo")
    print("=" * 40)
    
    generator = DemoImageGenerator()
    
    # Load sample prompts
    prompt_file = Path("prompts/sample_prompts.txt")
    if prompt_file.exists():
        with open(prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [
            "a beautiful sunset over mountains with a lake in the foreground",
            "a futuristic city with flying cars and neon lights",
            "a cozy coffee shop interior with warm lighting"
        ]
    
    print(f"📝 Processing {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts):
        print(f"\n🎨 [{i+1}/{len(prompts)}] Generating: '{prompt}'")
        
        # Generate image
        images = generator.generate_image(prompt, num_images=1)
        
        # Save image
        filename = f"batch_demo_{i+1:02d}"
        saved_paths = generator.save_image(images, filename)
        
        print(f"✅ Completed: {saved_paths[0]}")
    
    print(f"\n🎉 Batch demo completed! Generated {len(prompts)} images.")

def show_system_info():
    """Show system information and capabilities."""
    print("🎨 Task-02: Image Generation System Information")
    print("=" * 50)
    
    print("\n📋 System Capabilities:")
    print("  ✅ Text-to-image generation")
    print("  ✅ Multiple model support (Stable Diffusion, DALL-E-mini)")
    print("  ✅ Interactive command-line interface")
    print("  ✅ Batch image generation")
    print("  ✅ Parameter control (guidance scale, steps, etc.)")
    print("  ✅ Automatic image saving and organization")
    print("  ✅ Prompt management system")
    
    print("\n🤖 Supported Models:")
    print("  🎨 Stable Diffusion - High quality, detailed images")
    print("  ⚡ DALL-E-mini - Fast generation, good quality")
    
    print("\n⚙️ Generation Parameters:")
    print("  📏 Image size: 256x256 to 1024x1024")
    print("  🎯 Guidance scale: 1.0-20.0 (prompt adherence)")
    print("  🔄 Inference steps: 10-100 (quality vs speed)")
    print("  🌱 Random seed: Reproducible results")
    print("  🚫 Negative prompts: Avoid unwanted elements")
    
    print("\n📁 File Structure:")
    print("  📂 generated_images/ - Output directory")
    print("  📂 models/ - Downloaded model cache")
    print("  📂 prompts/ - Sample and custom prompts")
    
    print("\n🚀 Quick Start:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run interactive: python interactive_generator.py")
    print("  3. Generate image: python interactive_generator.py --prompt 'your prompt'")
    print("  4. Run test: python interactive_generator.py test")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Task-02 Demo: Image Generation")
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
        print("🎨 Task-02 Demo: Image Generation with Pre-trained Models")
        print("=" * 60)
        print("Available modes:")
        print("  interactive - Run interactive demo")
        print("  batch - Run batch generation demo")
        print("  info - Show system information")
        print("\nExample: python demo.py interactive")

