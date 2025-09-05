# Task-04: Image-to-Image Translation with cGAN

This project implements a complete image-to-image translation system using conditional Generative Adversarial Networks (cGAN) based on the pix2pix architecture. The system can transform images from one domain to another, providing an interactive interface for creative image translation.

## 🎯 Project Overview

The goal of this task is to:
- Implement an image-to-image translation model using conditional GANs (cGAN)
- Create a pix2pix architecture for paired image translation
- Provide an interactive interface for image transformation
- Support multiple image translation tasks (sketch-to-photo, day-to-night, etc.)
- Enable real-time image processing and batch operations

## 🚀 Features

- **Pix2Pix Architecture**: Complete implementation of conditional GAN for image translation
- **Multiple Translation Tasks**: Support for various image-to-image transformations
- **Interactive Interface**: User-friendly command-line interface
- **Real-time Processing**: Live image transformation capabilities
- **Batch Processing**: Process multiple images simultaneously
- **Model Management**: Save and load trained models
- **Data Augmentation**: Built-in data preprocessing and augmentation
- **Quality Control**: Configurable generation parameters and quality metrics

## 📋 Requirements

- Python 3.7+
- PyTorch for deep learning framework
- Torchvision for image processing
- NumPy for numerical operations
- Pillow (PIL) for image handling
- Matplotlib for visualization
- OpenCV for advanced image processing
- TQDM for progress tracking

## 🛠️ Installation

1. **Clone or download this project**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python interactive_translator.py test
   ```

## 📁 Project Structure

```
├── pix2pix_translator.py      # Main pix2pix implementation
├── interactive_translator.py  # Interactive interface
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/                      # Training and test data
├── models/                    # Saved cGAN models
├── generated_images/          # Translated image outputs
└── sample_data/              # Sample image pairs
```

## 🎮 Usage

### Quick Start

1. **Run the interactive interface**:
   ```bash
   python interactive_translator.py
   ```

2. **Translate an image**:
   ```
   translate sample_data/sketch.jpg
   ```

3. **Adjust settings**:
   ```
   settings
   ```

### Programmatic Usage

```python
from pix2pix_translator import Pix2PixTranslator

# Initialize the translator
translator = Pix2PixTranslator(model_type="pix2pix")

# Load a pre-trained model
translator.load_model("models/sketch_to_photo.pth")

# Translate an image
input_image = "sample_data/sketch.jpg"
output_image = translator.translate_image(
    input_path=input_image,
    output_path="generated_images/photo.jpg",
    translation_type="sketch_to_photo"
)

print(f"Image translated: {output_image}")
```

### Using Different Translation Types

```python
# Sketch to photo translation
translator = Pix2PixTranslator(translation_type="sketch_to_photo")

# Day to night translation
translator = Pix2PixTranslator(translation_type="day_to_night")

# Black and white to color
translator = Pix2PixTranslator(translation_type="bw_to_color")

# Custom translation task
translator = Pix2PixTranslator(translation_type="custom")
```

## ⚙️ Configuration

### Translation Parameters

- **translation_type**: Type of image translation task
- **image_size**: Input/output image dimensions (default: 256x256)
- **batch_size**: Number of images to process simultaneously
- **num_epochs**: Training epochs for model training
- **learning_rate**: Learning rate for training (default: 0.0002)
- **beta1**: Beta1 parameter for Adam optimizer (default: 0.5)

### Model Parameters

- **generator_type**: "unet" or "resnet" generator architecture
- **discriminator_type**: "patch" or "pixel" discriminator
- **lambda_l1**: L1 loss weight (default: 100.0)
- **lambda_gan**: GAN loss weight (default: 1.0)

## 📊 Translation Types

### Sketch-to-Photo
- **Input**: Hand-drawn sketches or line art
- **Output**: Realistic photographs
- **Use Cases**: Concept art, architectural visualization, fashion design

### Day-to-Night
- **Input**: Daytime photographs
- **Output**: Nighttime versions
- **Use Cases**: Real estate visualization, film production, urban planning

### Black-and-White to Color
- **Input**: Grayscale images
- **Output**: Colorized versions
- **Use Cases**: Historical photo restoration, artistic enhancement

### Style Transfer
- **Input**: Source images
- **Output**: Stylized versions
- **Use Cases**: Artistic effects, photo enhancement, creative projects

## 🔧 Customization

### Adding Custom Translation Tasks

1. Define your translation type in `pix2pix_translator.py`
2. Prepare paired training data
3. Configure model parameters for your specific task
4. Train the model on your dataset

### Modifying Model Architecture

Edit the parameters in `pix2pix_translator.py`:

```python
# In the __init__ method
translator = Pix2PixTranslator(
    translation_type="custom",
    generator_type="resnet",
    discriminator_type="patch",
    image_size=(512, 512),
    lambda_l1=200.0
)
```

### Using Different Image Sizes

```python
# Standard size (256x256)
translator = Pix2PixTranslator(image_size=(256, 256))

# High resolution (512x512)
translator = Pix2PixTranslator(image_size=(512, 512))

# Custom size
translator = Pix2PixTranslator(image_size=(384, 384))
```

## 🎯 Example Tasks

Try these example translation tasks:

- **Sketch to Photo**: Convert hand-drawn sketches to realistic photos
- **Day to Night**: Transform daytime scenes to nighttime
- **Black & White to Color**: Colorize grayscale images
- **Style Transfer**: Apply artistic styles to photos
- **Semantic to Real**: Convert semantic maps to realistic images

## 🚨 Troubleshooting

### Common Issues

1. **Poor Translation Quality**:
   - Increase training data size
   - Adjust model parameters
   - Use higher resolution images
   - Train for more epochs

2. **Slow Processing**:
   - Reduce image size
   - Use GPU acceleration
   - Optimize batch size
   - Enable model caching

3. **Memory Issues**:
   - Reduce batch size
   - Use smaller image dimensions
   - Enable gradient checkpointing
   - Use mixed precision training

### Performance Tips

- **Training Data**: Use high-quality, well-paired images
- **Model Architecture**: Choose appropriate generator/discriminator types
- **Hyperparameters**: Tune learning rates and loss weights
- **Hardware**: Use GPU for faster training and inference

## 📚 Technical Details

### Architecture

- **Generator**: U-Net or ResNet architecture for image generation
- **Discriminator**: PatchGAN discriminator for local patch discrimination
- **Loss Functions**: Combination of GAN loss and L1 reconstruction loss
- **Optimization**: Adam optimizer with specific hyperparameters

### Training Process

1. **Data Preparation**: Load and preprocess paired training images
2. **Model Initialization**: Set up generator and discriminator networks
3. **Training Loop**: Alternating generator and discriminator training
4. **Loss Calculation**: Compute GAN and L1 losses
5. **Model Validation**: Evaluate translation quality

### Inference Process

1. **Image Loading**: Load input image and preprocess
2. **Model Forward Pass**: Generate translated image
3. **Post-processing**: Apply final adjustments and enhancements
4. **Output Saving**: Save translated image to file

## 🤝 Contributing

Feel free to extend this project:

- Add support for other GAN architectures
- Implement additional translation tasks
- Create a web interface
- Add support for video translation
- Implement advanced loss functions

## 📄 License

This project is for educational purposes. Please respect the licenses of the underlying libraries.

## 🎉 Success!

You've successfully completed Task-04: Image-to-Image Translation with cGAN! The system can now:

✅ Implement pix2pix architecture for image translation  
✅ Perform various image-to-image transformations  
✅ Provide an interactive interface  
✅ Control translation parameters  
✅ Handle different translation tasks  

Happy image translating! 🎨











