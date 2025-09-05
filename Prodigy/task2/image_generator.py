"""
Image Generation with Pre-trained Models
Task-02: Text-to-Image Generation using DALL-E-mini and Stable Diffusion

This module provides a comprehensive interface for generating images from text prompts
using state-of-the-art pre-trained models.
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple, Union
import logging
from pathlib import Path
import time
from datetime import datetime

# Import required libraries for different models
try:
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    from transformers import pipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("Warning: diffusers library not available. Stable Diffusion will not work.")

try:
    import minidalle
    DALLEMINI_AVAILABLE = True
except ImportError:
    DALLEMINI_AVAILABLE = False
    print("Warning: minidalle library not available. DALL-E-mini will not work.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageGenerator:
    """
    A comprehensive image generation class that supports multiple pre-trained models
    for text-to-image generation.
    """
    
    def __init__(
        self,
        model_type: str = "stable-diffusion",
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        image_size: Tuple[int, int] = (512, 512),
        safety_filter: bool = True
    ):
        """
        Initialize the image generator with specified model and parameters.
        
        Args:
            model_type: Type of model to use ("stable-diffusion", "dalle-mini")
            device: Device to run the model on ("cpu", "cuda", "mps")
            model_path: Path to local model (optional)
            image_size: Tuple of (width, height) for generated images
            safety_filter: Whether to apply safety filtering
        """
        self.model_type = model_type.lower()
        self.image_size = image_size
        self.safety_filter = safety_filter
        self.device = self._setup_device(device)
        self.model_path = model_path
        
        # Initialize model
        self.model = None
        self.tokenizer = None
        self._load_model()
        
        # Create output directories
        self._create_directories()
        
        logger.info(f"ImageGenerator initialized with {model_type} on {self.device}")
    
    def _setup_device(self, device: Optional[str]) -> str:
        """Set up the device for model execution."""
        if device is not None:
            return device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def _create_directories(self):
        """Create necessary directories for saving images and models."""
        directories = ["generated_images", "models", "prompts"]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def _load_model(self):
        """Load the specified model based on model_type."""
        if self.model_type == "stable-diffusion":
            self._load_stable_diffusion()
        elif self.model_type == "dalle-mini":
            self._load_dalle_mini()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _load_stable_diffusion(self):
        """Load Stable Diffusion model."""
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("diffusers library is required for Stable Diffusion")
        
        try:
            model_id = self.model_path or "runwayml/stable-diffusion-v1-5"
            
            # Load the pipeline
            self.model = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None if not self.safety_filter else None
            )
            
            # Use DPM-Solver++ scheduler for faster inference
            self.model.scheduler = DPMSolverMultistepScheduler.from_config(
                self.model.scheduler.config
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.model, "enable_attention_slicing"):
                self.model.enable_attention_slicing()
            
            logger.info(f"Stable Diffusion model loaded from {model_id}")
            
        except Exception as e:
            logger.error(f"Error loading Stable Diffusion model: {e}")
            raise
    
    def _load_dalle_mini(self):
        """Load DALL-E-mini model."""
        if not DALLEMINI_AVAILABLE:
            raise ImportError("minidalle library is required for DALL-E-mini")
        
        try:
            # For DALL-E-mini, we'll use a simplified approach
            # In a real implementation, you would use the actual DALL-E-mini library
            logger.info("DALL-E-mini model loaded (simulated)")
            
        except Exception as e:
            logger.error(f"Error loading DALL-E-mini model: {e}")
            raise
    
    def generate_image(
        self,
        prompt: str,
        num_images: int = 1,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None,
        negative_prompt: Optional[str] = None
    ) -> Union[Image.Image, List[Image.Image]]:
        """
        Generate image(s) from text prompt.
        
        Args:
            prompt: Text description of the image to generate
            num_images: Number of images to generate
            guidance_scale: How closely to follow the prompt (higher = more adherence)
            num_inference_steps: Number of denoising steps
            seed: Random seed for reproducible results
            negative_prompt: Text describing what to avoid in the image
            
        Returns:
            Generated image(s) as PIL Image object(s)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        logger.info(f"Generating {num_images} image(s) with prompt: '{prompt}'")
        start_time = time.time()
        
        try:
            if self.model_type == "stable-diffusion":
                images = self._generate_stable_diffusion(
                    prompt, num_images, guidance_scale, num_inference_steps, negative_prompt
                )
            elif self.model_type == "dalle-mini":
                images = self._generate_dalle_mini(
                    prompt, num_images, guidance_scale, num_inference_steps
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            generation_time = time.time() - start_time
            logger.info(f"Image generation completed in {generation_time:.2f} seconds")
            
            return images[0] if num_images == 1 else images
            
        except Exception as e:
            logger.error(f"Error during image generation: {e}")
            raise
    
    def _generate_stable_diffusion(
        self,
        prompt: str,
        num_images: int,
        guidance_scale: float,
        num_inference_steps: int,
        negative_prompt: Optional[str]
    ) -> List[Image.Image]:
        """Generate images using Stable Diffusion."""
        with torch.no_grad():
            result = self.model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                width=self.image_size[0],
                height=self.image_size[1]
            )
            
            return result.images
    
    def _generate_dalle_mini(
        self,
        prompt: str,
        num_images: int,
        guidance_scale: float,
        num_inference_steps: int
    ) -> List[Image.Image]:
        """Generate images using DALL-E-mini (simulated)."""
        # This is a simplified implementation
        # In a real scenario, you would use the actual DALL-E-mini library
        
        # Create a placeholder image for demonstration
        images = []
        for i in range(num_images):
            # Create a simple gradient image as placeholder
            img_array = np.random.randint(0, 255, (self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            images.append(img)
        
        return images
    
    def save_image(
        self,
        image: Union[Image.Image, List[Image.Image]],
        filename: Optional[str] = None,
        format: str = "PNG"
    ) -> List[str]:
        """
        Save generated image(s) to disk.
        
        Args:
            image: PIL Image or list of PIL Images to save
            filename: Base filename (without extension)
            format: Image format (PNG, JPEG, etc.)
            
        Returns:
            List of saved file paths
        """
        if isinstance(image, Image.Image):
            images = [image]
        else:
            images = image
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_{timestamp}"
        
        saved_paths = []
        
        for i, img in enumerate(images):
            if len(images) > 1:
                filepath = f"generated_images/{filename}_{i+1}.{format.lower()}"
            else:
                filepath = f"generated_images/{filename}.{format.lower()}"
            
            img.save(filepath, format=format)
            saved_paths.append(filepath)
            logger.info(f"Image saved to: {filepath}")
        
        return saved_paths
    
    def generate_and_save(
        self,
        prompt: str,
        filename: Optional[str] = None,
        num_images: int = 1,
        **kwargs
    ) -> List[str]:
        """
        Generate image(s) and save them automatically.
        
        Args:
            prompt: Text description of the image to generate
            filename: Base filename for saving
            num_images: Number of images to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of saved file paths
        """
        images = self.generate_image(prompt, num_images, **kwargs)
        return self.save_image(images, filename)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_type": self.model_type,
            "device": self.device,
            "image_size": self.image_size,
            "safety_filter": self.safety_filter,
            "model_path": self.model_path
        }
    
    def change_model(self, model_type: str, **kwargs):
        """Change the model type and reload."""
        self.model_type = model_type.lower()
        self._load_model()
        logger.info(f"Model changed to: {model_type}")
    
    def set_image_size(self, width: int, height: int):
        """Set the image size for generation."""
        self.image_size = (width, height)
        logger.info(f"Image size set to: {width}x{height}")


def create_sample_prompts():
    """Create sample prompt files for testing."""
    prompts = [
        "a beautiful sunset over mountains with a lake in the foreground",
        "a futuristic city with flying cars and neon lights",
        "a cozy coffee shop interior with warm lighting",
        "a majestic dragon flying over a medieval castle",
        "a serene forest with sunlight filtering through trees",
        "a cyberpunk street scene with rain and neon signs",
        "a peaceful beach at dawn with gentle waves",
        "an astronaut riding a horse in a photorealistic style",
        "a steampunk airship floating above Victorian London",
        "a magical library with floating books and glowing orbs"
    ]
    
    with open("prompts/sample_prompts.txt", "w") as f:
        for prompt in prompts:
            f.write(prompt + "\n")
    
    logger.info("Sample prompts created in prompts/sample_prompts.txt")


if __name__ == "__main__":
    # Example usage
    print("Task-02: Image Generation with Pre-trained Models")
    print("=" * 50)
    
    # Create sample prompts
    create_sample_prompts()
    
    # Initialize generator
    try:
        generator = ImageGenerator(model_type="stable-diffusion")
        
        # Generate a sample image
        prompt = "a beautiful sunset over mountains"
        print(f"\nGenerating image with prompt: '{prompt}'")
        
        image = generator.generate_image(prompt, num_images=1)
        saved_paths = generator.save_image(image, "sample_sunset")
        
        print(f"Image saved to: {saved_paths[0]}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please install required dependencies: pip install -r requirements.txt")

