## Task-05: Neural Style Transfer (NST) – Brief

- Goal: Combine the content of one image with the style of another.
- Steps:
  1) Preprocess (resize, normalize, tensor)
  2) Extract VGG19 features
  3) Compute content/style/TV losses
  4) Optimize generated image
  5) Save and review outputs

Run single pair:
```bash
python interactive_translator.py --content "sample_data/content.jpg" --style "sample_data/style.jpg" --output "generated_images/stylized.jpg" --size 512 --iterations 300
```

Batch over folders:
```bash
python batch_nst.py --contents sample_data --styles sample_data --size 512 --iterations 300
```

References: Gatys et al. (CVPR 2016), PyTorch Tutorial, TensorFlow Hub.




