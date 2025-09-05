"""
Image-to-Image Translation with cGAN (Pix2Pix)
Task-04: Conditional Generative Adversarial Network for image translation

This module provides a comprehensive interface for image-to-image translation
using the pix2pix architecture with conditional GANs.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import time
import logging
from typing import Tuple, List, Optional, Dict, Any
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UNetGenerator(nn.Module):
    """
    U-Net Generator for pix2pix architecture.
    """
    
    def __init__(self, input_channels: int = 3, output_channels: int = 3, num_filters: int = 64):
        super(UNetGenerator, self).__init__()
        
        # Simple encoder-decoder architecture
        self.encoder = nn.Sequential(
            # Input: 3 -> 64
            nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 64 -> 128
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 128 -> 256
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 256 -> 512
            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.decoder = nn.Sequential(
            # 512 -> 256
            nn.ConvTranspose2d(num_filters * 8, num_filters * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.ReLU(inplace=True),
            
            # 256 -> 128
            nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.ReLU(inplace=True),
            
            # 128 -> 64
            nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            
            # 64 -> output_channels
            nn.ConvTranspose2d(num_filters, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for pix2pix architecture.
    """
    
    def __init__(self, input_channels: int = 6, num_filters: int = 64):
        super(PatchDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer
            nn.Conv2d(num_filters * 8, 1, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)


class Pix2PixDataset(Dataset):
    """
    Dataset for paired image training.
    """
    
    def __init__(self, data_dir: str, image_size: Tuple[int, int] = (256, 256), 
                 translation_type: str = "sketch_to_photo"):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.translation_type = translation_type
        
        # Find image pairs
        self.image_pairs = self._find_image_pairs()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        logger.info(f"Found {len(self.image_pairs)} image pairs for {translation_type}")
    
    def _find_image_pairs(self) -> List[Tuple[str, str]]:
        """Find paired images in the data directory."""
        pairs = []
        
        # Look for common naming patterns
        input_patterns = ["*_input.*", "*_source.*", "*_sketch.*", "*_day.*", "*_bw.*"]
        output_patterns = ["*_output.*", "*_target.*", "*_photo.*", "*_night.*", "*_color.*"]
        
        for input_pattern in input_patterns:
            input_files = list(self.data_dir.glob(input_pattern))
            for input_file in input_files:
                # Try to find corresponding output file
                base_name = input_file.stem.replace("_input", "").replace("_source", "").replace("_sketch", "").replace("_day", "").replace("_bw", "")
                
                for output_pattern in output_patterns:
                    output_file = self.data_dir / f"{base_name}_output{input_file.suffix}"
                    if output_file.exists():
                        pairs.append((str(input_file), str(output_file)))
                        break
        
        return pairs
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        input_path, output_path = self.image_pairs[idx]
        
        # Load images
        input_image = Image.open(input_path).convert('RGB')
        output_image = Image.open(output_path).convert('RGB')
        
        # Apply transforms
        input_tensor = self.transform(input_image)
        output_tensor = self.transform(output_image)
        
        return input_tensor, output_tensor


class Pix2PixTranslator:
    """
    Main pix2pix image translation system.
    """
    
    def __init__(
        self,
        translation_type: str = "sketch_to_photo",
        image_size: Tuple[int, int] = (256, 256),
        generator_type: str = "unet",
        discriminator_type: str = "patch",
        lambda_l1: float = 100.0,
        lambda_gan: float = 1.0,
        learning_rate: float = 0.0002,
        beta1: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Initialize the pix2pix translator.
        
        Args:
            translation_type: Type of translation task
            image_size: Input/output image dimensions
            generator_type: Generator architecture type
            discriminator_type: Discriminator architecture type
            lambda_l1: L1 loss weight
            lambda_gan: GAN loss weight
            learning_rate: Learning rate for training
            beta1: Beta1 parameter for Adam optimizer
            device: Device to use (cuda/cpu)
        """
        self.translation_type = translation_type
        self.image_size = image_size
        self.generator_type = generator_type
        self.discriminator_type = discriminator_type
        self.lambda_l1 = lambda_l1
        self.lambda_gan = lambda_gan
        self.learning_rate = learning_rate
        self.beta1 = beta1
        
        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.criterion_gan = None
        self.criterion_l1 = None
        
        self._initialize_models()
        
        # Create output directories
        self._create_directories()
        
        # Training statistics
        self.training_stats = {
            "epochs_trained": 0,
            "total_loss": 0.0,
            "g_loss": 0.0,
            "d_loss": 0.0,
            "training_time": 0.0
        }
        
        logger.info(f"Pix2PixTranslator initialized for {translation_type}")
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = ["data", "models", "generated_images", "sample_data"]
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def _initialize_models(self):
        """Initialize generator, discriminator, and optimizers."""
        # Generator
        if self.generator_type == "unet":
            self.generator = UNetGenerator(input_channels=3, output_channels=3).to(self.device)
        else:
            raise ValueError(f"Unsupported generator type: {self.generator_type}")
        
        # Discriminator
        if self.discriminator_type == "patch":
            self.discriminator = PatchDiscriminator(input_channels=6).to(self.device)
        else:
            raise ValueError(f"Unsupported discriminator type: {self.discriminator_type}")
        
        # Optimizers
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(self.beta1, 0.999))
        
        # Loss functions
        self.criterion_gan = nn.BCELoss().to(self.device)
        self.criterion_l1 = nn.L1Loss().to(self.device)
    
    def train(
        self,
        data_dir: str,
        num_epochs: int = 200,
        batch_size: int = 1,
        save_interval: int = 50,
        model_path: Optional[str] = None
    ):
        """
        Train the pix2pix model.
        
        Args:
            data_dir: Directory containing training data
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            save_interval: Interval for saving model checkpoints
            model_path: Path to save the trained model
        """
        logger.info(f"Starting training for {self.translation_type}")
        
        # Create dataset and dataloader
        dataset = Pix2PixDataset(data_dir, self.image_size, self.translation_type)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            
            for batch_idx, (input_images, target_images) in enumerate(dataloader):
                input_images = input_images.to(self.device)
                target_images = target_images.to(self.device)
                
                # Train discriminator
                self.d_optimizer.zero_grad()
                
                # Real images
                real_pairs = torch.cat([input_images, target_images], dim=1)
                real_labels = torch.ones(input_images.size(0), 1, 30, 30).to(self.device)
                real_output = self.discriminator(real_pairs)
                d_real_loss = self.criterion_gan(real_output, real_labels)
                
                # Fake images
                fake_images = self.generator(input_images)
                fake_pairs = torch.cat([input_images, fake_images.detach()], dim=1)
                fake_labels = torch.zeros(input_images.size(0), 1, 30, 30).to(self.device)
                fake_output = self.discriminator(fake_pairs)
                d_fake_loss = self.criterion_gan(fake_output, fake_labels)
                
                d_loss = (d_real_loss + d_fake_loss) * 0.5
                d_loss.backward()
                self.d_optimizer.step()
                
                # Train generator
                self.g_optimizer.zero_grad()
                
                fake_output = self.discriminator(fake_pairs)
                g_gan_loss = self.criterion_gan(fake_output, real_labels)
                g_l1_loss = self.criterion_l1(fake_images, target_images)
                g_loss = g_gan_loss * self.lambda_gan + g_l1_loss * self.lambda_l1
                
                g_loss.backward()
                self.g_optimizer.step()
                
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
            
            # Update statistics
            self.training_stats["epochs_trained"] += 1
            self.training_stats["g_loss"] = epoch_g_loss / len(dataloader)
            self.training_stats["d_loss"] = epoch_d_loss / len(dataloader)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}] - G Loss: {self.training_stats['g_loss']:.4f}, D Loss: {self.training_stats['d_loss']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_model(f"models/{self.translation_type}_checkpoint_{epoch+1}.pth")
        
        # Save final model
        if model_path:
            self.save_model(model_path)
        else:
            self.save_model(f"models/{self.translation_type}_final.pth")
        
        self.training_stats["training_time"] = time.time() - start_time
        logger.info(f"Training completed in {self.training_stats['training_time']:.2f} seconds")
    
    def translate_image(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        translation_type: Optional[str] = None
    ) -> str:
        """
        Translate a single image.
        
        Args:
            input_path: Path to input image
            output_path: Path to save output image
            translation_type: Type of translation (overrides instance setting)
        
        Returns:
            Path to the generated image
        """
        if translation_type:
            self.translation_type = translation_type
        
        # Load and preprocess input image
        input_image = Image.open(input_path).convert('RGB')
        input_image = input_image.resize(self.image_size)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        input_tensor = transform(input_image).unsqueeze(0).to(self.device)
        
        # Generate output
        with torch.no_grad():
            output_tensor = self.generator(input_tensor)
        
        # Post-process output
        output_tensor = (output_tensor + 1) / 2.0  # Denormalize
        output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
        
        # Save output
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"generated_images/{self.translation_type}_{timestamp}.jpg"
        
        output_image.save(output_path)
        logger.info(f"Image translated: {output_path}")
        
        return output_path
    
    def translate_batch(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        translation_type: Optional[str] = None
    ) -> List[str]:
        """
        Translate multiple images in batch.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output images
            translation_type: Type of translation
        
        Returns:
            List of paths to generated images
        """
        if output_dir is None:
            output_dir = "generated_images"
        
        Path(output_dir).mkdir(exist_ok=True)
        
        input_paths = list(Path(input_dir).glob("*.jpg")) + list(Path(input_dir).glob("*.png"))
        output_paths = []
        
        logger.info(f"Translating {len(input_paths)} images...")
        
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
                logger.error(f"Error translating {input_path}: {e}")
        
        logger.info(f"Batch translation completed: {len(output_paths)} images")
        return output_paths
    
    def save_model(self, file_path: str):
        """Save the trained model."""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'translation_type': self.translation_type,
            'image_size': self.image_size,
            'training_stats': self.training_stats
        }, file_path)
        
        logger.info(f"Model saved to: {file_path}")
    
    def load_model(self, file_path: str):
        """Load a trained model."""
        checkpoint = torch.load(file_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        self.translation_type = checkpoint.get('translation_type', self.translation_type)
        self.image_size = checkpoint.get('image_size', self.image_size)
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        
        logger.info(f"Model loaded from: {file_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "translation_type": self.translation_type,
            "image_size": self.image_size,
            "generator_type": getattr(self, 'generator_type', 'unet'),
            "discriminator_type": getattr(self, 'discriminator_type', 'patch'),
            "lambda_l1": self.lambda_l1,
            "lambda_gan": self.lambda_gan,
            "device": self.device,
            "training_stats": self.training_stats
        }


def create_sample_data():
    """Create sample training data."""
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    # Create simple sample images
    sample_images = [
        ("sketch_input.jpg", "sketch_output.jpg"),
        ("day_input.jpg", "night_output.jpg"),
        ("bw_input.jpg", "color_output.jpg")
    ]
    
    for input_name, output_name in sample_images:
        # Create simple test images
        input_img = Image.new('RGB', (256, 256), color='white')
        output_img = Image.new('RGB', (256, 256), color='lightblue')
        
        input_img.save(sample_dir / input_name)
        output_img.save(sample_dir / output_name)
    
    logger.info(f"Sample data created in {sample_dir}")


if __name__ == "__main__":
    # Example usage
    print("Task-04: Image-to-Image Translation with cGAN")
    print("=" * 50)
    
    # Create sample data
    create_sample_data()
    
    # Initialize translator
    translator = Pix2PixTranslator(translation_type="sketch_to_photo")
    
    # Example translation
    if Path("sample_data/sketch_input.jpg").exists():
        output_path = translator.translate_image(
            "sample_data/sketch_input.jpg",
            "generated_images/sample_translation.jpg"
        )
        print(f"Sample translation completed: {output_path}")
    
    print("Pix2Pix system ready for use!")
