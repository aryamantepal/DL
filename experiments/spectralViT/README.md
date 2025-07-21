# SpectralViT for Indian Pines Hyperspectral Classification

Vision Transformer (ViT) model for hyperspectral image classification on the Indian Pines dataset.

## Key Features
- Pure Transformer architecture for spectral-spatial feature learning
- Patch embedding with positional encoding
- End-to-end training on hyperspectral patches
- Achieves ~95% accuracy on Indian Pines

## Dataset
Indian Pines Hyperspectral Dataset:
- 145×145 pixels, 220 spectral bands
- 16 ground truth classes
- Download from [Purdue University](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)

## Usage
1. Install requirements:
pip install torch scipy scikit-learn

2. Training
python main.py


### Model Architecture:
SpectralViT(
  (patch_embed): SpectralPatchEmbedding()
  (transformer_blocks): Sequential(TransformerBlock x4)
  (head): Linear()
)
