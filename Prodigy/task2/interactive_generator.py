"""
Interactive Image Generator Interface
Task-02: Command-line interface for text-to-image generation

This module provides an interactive command-line interface for generating images
from text prompts using pre-trained models.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from image_generator import ImageGenerator, create_sample_prompts

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractiveImageGenerator:
    """
    Interactive command-line interface for image generation.
    """
    
    def __init__(self):
        """Initialize the interactive interface."""
        self.generator = None
        self.current_model = "stable-diffusion"
        self.current_settings = {
            "num_images": 1,
            "guidance_scale": 7.5,
            "num_inference_steps": 50,
            "image_size": (512, 512),
            "seed": None,
            "negative_prompt": None
        }
        self.prompts = []
        self.current_prompt_index = 0
        
        # Load sample prompts
        self._load_sample_prompts()
        
        print("🎨 Task-02: Image Generation with Pre-trained Models")
        print("=" * 60)
        print("Type 'help' for available commands")
        print("Type 'quit' to exit")
        print()
    
    def _load_sample_prompts(self):
        """Load sample prompts from file."""
        prompt_file = Path("prompts/sample_prompts.txt")
        if prompt_file.exists():
            with open(prompt_file, "r") as f:
                self.prompts = [line.strip() for line in f if line.strip()]
        else:
            # Create sample prompts if they don't exist
            create_sample_prompts()
            self._load_sample_prompts()
    
    def initialize_generator(self, model_type: str = "stable-diffusion"):
        """Initialize the image generator with specified model."""
        try:
            if self.generator is None or model_type != self.current_model:
                print(f"🔄 Initializing {model_type} model...")
                self.generator = ImageGenerator(
                    model_type=model_type,
                    image_size=self.current_settings["image_size"]
                )
                self.current_model = model_type
                print(f"✅ {model_type} model initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Error initializing {model_type} model: {e}")
            return False
    
    def run(self):
        """Run the interactive interface."""
        while True:
            try:
                command = input("\n🎨 ImageGen> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    print("👋 Goodbye!")
                    break
                elif command == "help":
                    self.show_help()
                elif command == "generate":
                    self.generate_image_interactive()
                elif command == "settings":
                    self.show_settings()
                elif command == "change_model":
                    self.change_model_interactive()
                elif command == "list_prompts":
                    self.list_prompts()
                elif command == "load_prompts":
                    self.load_prompts_interactive()
                elif command == "next_prompt":
                    self.next_prompt()
                elif command == "test":
                    self.run_test()
                elif command == "info":
                    self.show_info()
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
🎨 Available Commands:

📝 Generation:
  generate              - Generate image with interactive prompt input
  generate <prompt>     - Generate image with specified prompt
  next_prompt          - Use next prompt from loaded list

⚙️ Settings & Configuration:
  settings             - Show current generation settings
  change_model         - Change the model type
  info                 - Show model and system information

📋 Prompt Management:
  list_prompts         - Show all loaded prompts
  load_prompts         - Load prompts from file

🧪 Testing:
  test                 - Run a quick test generation
  help                 - Show this help message
  quit                 - Exit the program

💡 Tips:
  - Use descriptive prompts for better results
  - Adjust guidance_scale for prompt adherence
  - Increase num_inference_steps for higher quality
  - Use negative_prompt to avoid unwanted elements
        """
        print(help_text)
    
    def generate_image_interactive(self):
        """Generate image with interactive prompt input."""
        if not self.initialize_generator(self.current_model):
            return
        
        print("\n📝 Enter your image description:")
        prompt = input("Prompt: ").strip()
        
        if not prompt:
            print("❌ No prompt provided.")
            return
        
        self.generate_with_prompt(prompt)
    
    def generate_with_prompt(self, prompt: str):
        """Generate image with the given prompt."""
        if not self.initialize_generator(self.current_model):
            return
        
        print(f"\n🎨 Generating image with prompt: '{prompt}'")
        print(f"📊 Settings: {self.current_settings['num_images']} image(s), "
              f"guidance_scale={self.current_settings['guidance_scale']}, "
              f"steps={self.current_settings['num_inference_steps']}")
        
        try:
            # Generate image
            images = self.generator.generate_image(
                prompt=prompt,
                num_images=self.current_settings["num_images"],
                guidance_scale=self.current_settings["guidance_scale"],
                num_inference_steps=self.current_settings["num_inference_steps"],
                seed=self.current_settings["seed"],
                negative_prompt=self.current_settings["negative_prompt"]
            )
            
            # Save image
            filename = self._create_filename_from_prompt(prompt)
            saved_paths = self.generator.save_image(images, filename)
            
            print(f"✅ Image(s) generated successfully!")
            for path in saved_paths:
                print(f"📁 Saved to: {path}")
            
        except Exception as e:
            print(f"❌ Error generating image: {e}")
    
    def _create_filename_from_prompt(self, prompt: str) -> str:
        """Create a filename from the prompt."""
        # Clean the prompt for filename
        clean_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_prompt = clean_prompt.replace(' ', '_')[:30]  # Limit length
        return f"generated_{clean_prompt}"
    
    def show_settings(self):
        """Show current generation settings."""
        print("\n⚙️ Current Settings:")
        print(f"  Model: {self.current_model}")
        print(f"  Number of images: {self.current_settings['num_images']}")
        print(f"  Guidance scale: {self.current_settings['guidance_scale']}")
        print(f"  Inference steps: {self.current_settings['num_inference_steps']}")
        print(f"  Image size: {self.current_settings['image_size'][0]}x{self.current_settings['image_size'][1]}")
        print(f"  Seed: {self.current_settings['seed'] or 'Random'}")
        print(f"  Negative prompt: {self.current_settings['negative_prompt'] or 'None'}")
        
        # Ask if user wants to change settings
        change = input("\nWould you like to change settings? (y/n): ").strip().lower()
        if change == 'y':
            self.change_settings_interactive()
    
    def change_settings_interactive(self):
        """Change settings interactively."""
        print("\n🔧 Change Settings:")
        
        # Number of images
        try:
            num_images = input(f"Number of images (current: {self.current_settings['num_images']}): ").strip()
            if num_images:
                self.current_settings['num_images'] = int(num_images)
        except ValueError:
            print("❌ Invalid number of images. Keeping current setting.")
        
        # Guidance scale
        try:
            guidance = input(f"Guidance scale (current: {self.current_settings['guidance_scale']}): ").strip()
            if guidance:
                self.current_settings['guidance_scale'] = float(guidance)
        except ValueError:
            print("❌ Invalid guidance scale. Keeping current setting.")
        
        # Inference steps
        try:
            steps = input(f"Inference steps (current: {self.current_settings['num_inference_steps']}): ").strip()
            if steps:
                self.current_settings['num_inference_steps'] = int(steps)
        except ValueError:
            print("❌ Invalid inference steps. Keeping current setting.")
        
        # Image size
        try:
            size_input = input(f"Image size WxH (current: {self.current_settings['image_size'][0]}x{self.current_settings['image_size'][1]}): ").strip()
            if size_input and 'x' in size_input:
                width, height = map(int, size_input.split('x'))
                self.current_settings['image_size'] = (width, height)
                if self.generator:
                    self.generator.set_image_size(width, height)
        except ValueError:
            print("❌ Invalid image size. Keeping current setting.")
        
        # Seed
        try:
            seed_input = input(f"Random seed (current: {self.current_settings['seed'] or 'Random'}): ").strip()
            if seed_input:
                if seed_input.lower() == 'none':
                    self.current_settings['seed'] = None
                else:
                    self.current_settings['seed'] = int(seed_input)
        except ValueError:
            print("❌ Invalid seed. Keeping current setting.")
        
        # Negative prompt
        negative = input(f"Negative prompt (current: {self.current_settings['negative_prompt'] or 'None'}): ").strip()
        if negative:
            self.current_settings['negative_prompt'] = negative
        elif negative.lower() == 'none':
            self.current_settings['negative_prompt'] = None
        
        print("✅ Settings updated!")
    
    def change_model_interactive(self):
        """Change model type interactively."""
        print("\n🤖 Available Models:")
        print("  1. stable-diffusion (High quality, slower)")
        print("  2. dalle-mini (Faster, good quality)")
        
        choice = input("Select model (1-2): ").strip()
        
        if choice == "1":
            new_model = "stable-diffusion"
        elif choice == "2":
            new_model = "dalle-mini"
        else:
            print("❌ Invalid choice. Keeping current model.")
            return
        
        if new_model != self.current_model:
            self.current_model = new_model
            self.generator = None  # Force reinitialization
            print(f"✅ Model changed to: {new_model}")
        else:
            print("✅ Already using selected model.")
    
    def list_prompts(self):
        """List all loaded prompts."""
        if not self.prompts:
            print("📝 No prompts loaded.")
            return
        
        print(f"\n📝 Loaded Prompts ({len(self.prompts)} total):")
        for i, prompt in enumerate(self.prompts):
            marker = " → " if i == self.current_prompt_index else "   "
            print(f"{marker}{i+1:2d}. {prompt}")
    
    def load_prompts_interactive(self):
        """Load prompts from file interactively."""
        print("\n📁 Load Prompts from File:")
        print("Available prompt files:")
        
        prompt_dir = Path("prompts")
        if prompt_dir.exists():
            prompt_files = list(prompt_dir.glob("*.txt"))
            for i, file in enumerate(prompt_files):
                print(f"  {i+1}. {file.name}")
        
        filename = input("Enter filename (or press Enter for sample_prompts.txt): ").strip()
        if not filename:
            filename = "sample_prompts.txt"
        
        filepath = prompt_dir / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                self.prompts = [line.strip() for line in f if line.strip()]
            self.current_prompt_index = 0
            print(f"✅ Loaded {len(self.prompts)} prompts from {filename}")
        else:
            print(f"❌ File not found: {filename}")
    
    def next_prompt(self):
        """Use the next prompt from the loaded list."""
        if not self.prompts:
            print("📝 No prompts loaded. Use 'load_prompts' first.")
            return
        
        if self.current_prompt_index >= len(self.prompts):
            self.current_prompt_index = 0
            print("🔄 Reached end of prompts, starting over...")
        
        prompt = self.prompts[self.current_prompt_index]
        print(f"\n📝 Using prompt {self.current_prompt_index + 1}/{len(self.prompts)}: '{prompt}'")
        
        self.generate_with_prompt(prompt)
        self.current_prompt_index += 1
    
    def run_test(self):
        """Run a quick test generation."""
        print("\n🧪 Running Test Generation...")
        
        if not self.initialize_generator(self.current_model):
            return
        
        test_prompt = "a simple red circle on a white background"
        print(f"Test prompt: '{test_prompt}'")
        
        try:
            # Use minimal settings for quick test
            images = self.generator.generate_image(
                prompt=test_prompt,
                num_images=1,
                guidance_scale=7.5,
                num_inference_steps=20  # Fewer steps for faster test
            )
            
            saved_paths = self.generator.save_image(images, "test_generation")
            print(f"✅ Test successful! Image saved to: {saved_paths[0]}")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
    
    def show_info(self):
        """Show model and system information."""
        print("\nℹ️ System Information:")
        
        if self.generator:
            info = self.generator.get_model_info()
            print(f"  Model type: {info['model_type']}")
            print(f"  Device: {info['device']}")
            print(f"  Image size: {info['image_size']}")
            print(f"  Safety filter: {info['safety_filter']}")
        else:
            print("  Model: Not initialized")
        
        print(f"  Current model: {self.current_model}")
        print(f"  Loaded prompts: {len(self.prompts)}")
        print(f"  Current prompt index: {self.current_prompt_index}")


def main():
    """Main function to run the interactive interface."""
    parser = argparse.ArgumentParser(description="Interactive Image Generator")
    parser.add_argument("command", nargs="?", help="Command to run")
    parser.add_argument("--model", default="stable-diffusion", help="Model type to use")
    parser.add_argument("--prompt", help="Text prompt for image generation")
    
    args = parser.parse_args()
    
    if args.command == "test":
        # Run test mode
        generator = InteractiveImageGenerator()
        generator.run_test()
    elif args.prompt:
        # Generate image with provided prompt
        generator = InteractiveImageGenerator()
        if generator.initialize_generator(args.model):
            generator.generate_with_prompt(args.prompt)
    else:
        # Run interactive mode
        generator = InteractiveImageGenerator()
        generator.run()


if __name__ == "__main__":
    main()

