# Neural Style Transfer – Short Report

## Methodology
- VGG19 features (content: conv4_2; style: conv1_1..conv5_1)
- Losses: content MSE, style Gram MSE, total variation
- Optimize pixels with Adam for N iterations

## Results
- Default: image_size=512, iterations=300, style_weight=1e6
- Larger size improves detail; higher style_weight increases stylization

## Challenges
- Memory/time at high resolutions; GPU recommended

## Reproduce
```bash
python interactive_translator.py --content sample_data/content.jpg --style sample_data/style.jpg --output generated_images/stylized.jpg --size 512 --iterations 300
```

## References
1. Gatys et al., CVPR 2016
2. PyTorch Tutorial – Neural Style Transfer
3. TensorFlow Hub – Style Transfer


