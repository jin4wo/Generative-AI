# Task-02: Image Generation with Pre-trained Models

This project implements a complete image generation system using pre-trained generative models like DALL-E-mini and Stable Diffusion. The system can generate high-quality images from text prompts, providing an interactive interface for creative image generation.

## 🎯 Project Overview

The goal of this task is to:
- Generate images from text descriptions using pre-trained models
- Provide an interactive interface for image generation
- Support multiple pre-trained models (DALL-E-mini, Stable Diffusion)
- Create a user-friendly system for creative image generation
- Evaluate and compare different model outputs

## 🚀 Features

- **Multiple Models**: Support for DALL-E-mini and Stable Diffusion
- **Text-to-Image Generation**: Create images from natural language descriptions
- **Interactive Interface**: User-friendly command-line interface
- **Image Quality Control**: Adjustable parameters for generation quality
- **Batch Generation**: Generate multiple images from the same prompt
- **Image Saving**: Automatic saving with descriptive filenames
- **Model Comparison**: Compare outputs from different models

## 📋 Requirements

- Python 3.7+
- PyTorch
- Diffusers library (for Stable Diffusion)
- Transformers library
- Pillow (PIL)
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
├── image_generator.py          # Main image generation implementation
├── interactive_generator.py    # Interactive interface
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── generated_images/           # Generated images (created after generation)
├── models/                     # Downloaded models (created after first use)
└── prompts/                    # Sample text prompts
```

## 🎮 Usage

### Quick Start

1. **Run the interactive interface**:
   ```bash
   python interactive_generator.py
   ```

2. **Generate an image**:
   ```
   generate a beautiful sunset over mountains
   ```

3. **Adjust settings**:
   ```
   settings
   ```

### Programmatic Usage

```python
from image_generator import ImageGenerator

# Initialize the generator
generator = ImageGenerator(model_type="dalle-mini")

# Generate image
prompt = "a beautiful sunset over mountains"
image = generator.generate_image(
    prompt=prompt,
    num_images=1,
    guidance_scale=7.5,
    num_inference_steps=50
)

# Save the image
generator.save_image(image, "sunset_mountains.png")
```

### Using Different Models

```python
# Use DALL-E-mini
generator = ImageGenerator(model_type="dalle-mini")

# Use Stable Diffusion
generator = ImageGenerator(model_type="stable-diffusion")

# Use Stable Diffusion with different variants
generator = ImageGenerator(model_type="stable-diffusion-v1-5")
```

## ⚙️ Configuration

### Generation Parameters

- **num_images**: Number of images to generate (default: 1)
- **guidance_scale**: How closely to follow the prompt (default: 7.5)
- **num_inference_steps**: Number of denoising steps (default: 50)
- **height/width**: Image dimensions (default: 512x512)
- **seed**: Random seed for reproducible results

### Model Parameters

- **model_type**: Choose between "dalle-mini" and "stable-diffusion"
- **device**: CPU or CUDA device for generation
- **safety_filter**: Enable/disable content filtering

## 📊 Model Comparison

### DALL-E-mini
- **Pros**: Fast generation, good quality, easy to use
- **Cons**: Limited resolution, less detailed than Stable Diffusion
- **Best for**: Quick prototypes, simple scenes, artistic styles

### Stable Diffusion
- **Pros**: High quality, detailed images, highly customizable
- **Cons**: Slower generation, requires more computational resources
- **Best for**: Professional quality, detailed scenes, specific styles

## 🔧 Customization

### Adding Custom Prompts

1. Create a file in `prompts/` directory
2. Add your text prompts (one per line)
3. Use the `load_prompts` command in the interactive interface

### Modifying Generation Parameters

Edit the parameters in `image_generator.py`:

```python
# In the generate_image method
image = generator.generate_image(
    prompt=prompt,
    num_images=2,              # Generate 2 images
    guidance_scale=10.0,       # Stronger prompt adherence
    num_inference_steps=100    # More steps for better quality
)
```

### Using Different Model Variants

```python
# Stable Diffusion v1.5
generator = ImageGenerator(model_type="stable-diffusion-v1-5")

# Stable Diffusion v2.1
generator = ImageGenerator(model_type="stable-diffusion-v2-1")

# DALL-E-mini with different sizes
generator = ImageGenerator(model_type="dalle-mini", image_size="1024x1024")
```

## 🎯 Example Prompts

Try these example prompts to test the system:

- "a beautiful sunset over mountains with a lake in the foreground"
- "a futuristic city with flying cars and neon lights"
- "a cozy coffee shop interior with warm lighting"
- "a majestic dragon flying over a medieval castle"
- "a serene forest with sunlight filtering through trees"
- "a cyberpunk street scene with rain and neon signs"
- "a peaceful beach at dawn with gentle waves"

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce image resolution
   - Use CPU instead of GPU
   - Close other applications to free memory
   - Use DALL-E-mini instead of Stable Diffusion

2. **Slow Generation**:
   - Use GPU if available (CUDA)
   - Reduce num_inference_steps
   - Use DALL-E-mini for faster results
   - Generate smaller images

3. **Poor Quality Output**:
   - Increase num_inference_steps
   - Adjust guidance_scale
   - Use more descriptive prompts
   - Try different models

### Performance Tips

- **GPU Usage**: Models automatically use CUDA if available
- **Batch Generation**: Generate multiple images at once for efficiency
- **Model Caching**: Models are downloaded once and cached locally
- **Prompt Engineering**: Use detailed, specific descriptions for better results

## 📚 Technical Details

### Architecture

- **DALL-E-mini**: Transformer-based model with discrete VAE
- **Stable Diffusion**: Latent diffusion model with CLIP text encoder
- **Image Processing**: PIL for image handling and saving
- **Model Loading**: Hugging Face transformers and diffusers libraries

### Generation Process

1. **Text Encoding**: Prompt is encoded using CLIP text encoder
2. **Latent Generation**: Model generates image in latent space
3. **Decoding**: Latent representation is decoded to pixel space
4. **Post-processing**: Image is resized, filtered, and saved

## 🤝 Contributing

Feel free to extend this project:

- Add support for other image generation models
- Implement image editing capabilities
- Create a web interface
- Add support for different image formats
- Implement prompt optimization techniques

## 📄 License

This project is for educational purposes. Please respect the licenses of the underlying libraries and models.

## 🎉 Success!

You've successfully completed Task-02: Image Generation with Pre-trained Models! The system can now:

✅ Generate images from text prompts  
✅ Use multiple pre-trained models  
✅ Provide an interactive interface  
✅ Control generation parameters  
✅ Save and manage generated images  

Happy image generating! 🎨

