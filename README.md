# Indian Textile Pattern Symmetry Classification and Generation

An end-to-end deep learning pipeline that classifies, augments, and generates traditional Indian textile patterns using mathematical symmetry theory and conditional generative adversarial networks.

## Overview

Indian textiles are mathematically structured : a Patola silk has specific rotational and mirror symmetry properties dictated by the double ikat weaving technique, while a Kancheepuram check follows a different symmetry group entirely. This project uses wallpaper group theory (the 17 mathematically distinct plane symmetry groups) to classify these structures, augment a small dataset in a symmetry-preserving way, and train a conditional GAN to generate new textile-inspired patterns per class.

The project covers 10 traditional Indian textile categories: Ajrakh, Bagru, Ikat, Kancheepuram Checks, Leheriya, Madras Plaids, Manipuri Phanke, Mizopuan, Patola, and Sanganeri.


## Notebooks

### 1. Textile Classifier

This notebook builds a symmetry-aware classifier for Indian textile patterns. It extracts features using FFT-based tile detection, colour statistics, texture analysis, and symmetry scores, then trains an SVM to classify each image into one of the 10 textile categories. It also classifies each image into one of the 17 wallpaper groups and outputs a results CSV used by the augmentation pipeline.

Key components: FFT tile detection, wallpaper group classification, SVM with cross-validation, confusion matrix and per-class accuracy.

### 2. Textile Augmentation and Normalised Conditional DCGAN

This notebook has two parts. The first part uses the wallpaper group classifications from the classifier to apply only mathematically valid augmentation transforms to each image. For example, a pattern classified as pmm (vertical and horizontal mirror symmetry) is augmented with horizontal flips, vertical flips, and 180 degree rotations, while a p1 pattern (no symmetry) only receives brightness and contrast adjustments. This expands the dataset from 49 to 593 images while preserving structural integrity.

The second part trains a conditional DCGAN with spectral normalisation on the augmented dataset. The Generator takes a noise vector and a class label as input and learns to produce textile-inspired images per class at 128x128 resolution. Spectral normalisation is applied to both Generator and Discriminator layers to stabilise training.

## Dataset

The dataset contains 25 images per class across 10 Indian textile categories, organised into class subfolders. The ZIP file is to be uploaded directly in the notebooks.
Dataset Link: [Indian Textile Dataset](https://huggingface.co/datasets/shreyaph16/Indian_Textile_Dataset/tree/main)

## How to Run

### Requirements

- Google Colab with GPU enabled (Runtime > Change runtime type > T4 GPU)
- Google Drive for saving augmented images and model checkpoints
- The dataset ZIP file included in this repository

### Step 1: Run the Classifier

1. Open the Textile Classifier in Google Colab
2. Run the first cell to install dependencies
3. Upload dataset when prompted
4. Run all cells in order
5. Download upgraded_results.csv when it is generated — you will need this for the augmentation notebook

### Step 2: Run Augmentation and GAN Training

1. Open Textile_Augmentation_NormalisedDCGAN notebooks in Google Colab
2. Upload the Textile Dataset when prompted in the first upload cell
3. Upload upgraded_results.csv when prompted in the second upload cell
4. Mount Google Drive when prompted — augmented images and model checkpoints are saved there
5. Run all cells in order
6. Training runs for 200 epochs and visualises one generated image per class every 10 epochs

## Key Technical Decisions

**Symmetry-constrained augmentation** applies transforms based on the mathematical symmetry group of each pattern rather than applying random transforms blindly. This preserves structural integrity in the augmented dataset and is the main methodological contribution of this project.

**Conditional generation** feeds the class label into both the Generator and Discriminator so the model learns each textile class separately rather than blending all 10 classes into one undifferentiated style.

**Spectral normalisation** wraps each convolutional layer to keep weight matrices in a controlled range throughout training, reducing the instability (loss spikes, quality degradation between epochs) that is common when training GANs on small datasets.

## Limitations and Known Issues

The dataset size of 593 augmented images is small for GAN training. Generated images show class-differentiated colour palettes and rough texture, but lack the geometric precision of the original textiles. Increasing to 500 raw images per class would significantly improve output quality.

StyleGAN2-ADA was evaluated as an alternative architecture but encountered compatibility issues with Python 3.12 and NumPy 2.0 in the current Colab environment. This is documented in case future contributors want to attempt it with a compatible environment.

GAN training is inherently unstable and peak visual quality does not always occur at the final epoch. 

## Future Work

- Expand dataset to 500 images per class
- Implement WGAN-GP loss for more stable training
- Add self-attention layers to the Generator for better long-range geometric pattern learning
- Run generated images through the symmetry classifier to evaluate structural preservation
- Attempt StyleGAN2-ADA fine-tuning in a compatible environment

## Acknowledgements

This project was developed as part of a research exploration into the intersection of crystallographic symmetry theory, Indian textile heritage, and generative deep learning.
