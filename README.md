# Hybrid ConvNeXt-Tiny–Vision Mamba with Hemispheric Difference-Aware Attention and Bioinspired Optimization for Stroke Classification

This repository contains the implementation and experimental reproduction of a hybrid deep learning framework for binary stroke classification from rendered axial CT slice images.

The proposed framework combines:

- ConvNeXt-Tiny for hierarchical local feature extraction
- Vision Mamba for long-range contextual feature learning
- Adaptive Hemispheric Difference Attention (AHDA)
- Adaptive Cross-Attention Fusion (ACAF)
- Adaptive Lévy Flight Gorilla Troops Optimizer (ALGTO)
- Fully connected binary classification

The main objective is to jointly capture local visual features, long-range contextual dependencies, interhemispheric asymmetry, and complementary information between local and contextual representations.

---

## 1. Project Overview

Stroke requires rapid and accurate diagnosis, yet automated interpretation of computed tomography (CT) images remains challenging because of subtle lesion characteristics, anatomical variability, and similarities between normal and pathological tissue patterns.

This study presents a hybrid ConvNeXt-Tiny–Vision Mamba framework for binary stroke classification from rendered axial CT slice images.

The proposed framework combines hierarchical local feature extraction using ConvNeXt-Tiny, long-range contextual modeling using Vision Mamba, interhemispheric difference modeling using AHDA, cross-attention feature integration using ACAF, and hyperparameter optimization using ALGTO.

The overall framework is:

    Brain CT Image
           |
           v
    Image Preprocessing
           |
     +-----+-----+
     |           |
     v           v
ConvNeXt-Tiny  Vision Mamba
     |           |
     v           v
Local Features  Contextual Features
     |           |
     +-----+-----+
           |
           v
          AHDA
           |
           v
Hemispheric Difference
Enhanced Features
           |
           v
          ACAF
           |
           v
Cross-Attention Fusion
           |
           v
Global Average Pooling
           |
           v
         Dropout
           |
           v
Fully Connected Layer
           |
           v
        Sigmoid
           |
           v
  Stroke Probability
           ^
           |
         ALGTO
           |
Hyperparameter Optimization

---

## 2. Proposed Architecture

The proposed architecture consists of two complementary feature extraction branches.

### ConvNeXt-Tiny Branch

ConvNeXt-Tiny is used to extract hierarchical local spatial features from the CT images.

The model uses ImageNet-1K pretrained weights, and all layers are fine-tuned.

The four stages produce:

| Stage | Feature Map |
|---|---|
| Stage 1 | 56 × 56 × 96 |
| Stage 2 | 28 × 28 × 192 |
| Stage 3 | 14 × 14 × 384 |
| Stage 4 | 7 × 7 × 768 |

The proposed framework uses the **Stage 3** representation:

    14 × 14 × 384

Stage 3 is selected instead of Stage 4 because its finer spatial resolution is required for the hemispheric comparison performed by AHDA.

The ConvNeXt-Tiny stages contain:

    3 + 3 + 9 + 3 ConvNeXt blocks

### Vision Mamba Branch

Vision Mamba is used to capture long-range contextual dependencies.

The Vision Mamba branch is trained from scratch with all layers trainable.

Patch embedding is performed using:

    4 × 4 convolution
    Stride = 4

This produces:

    56 × 56 = 3,136 tokens

with embedding dimension:

    384

The branch contains:

    24 Vision Mamba blocks
    State dimension = 16

The output is reshaped to:

    56 × 56 × 384

and adaptively average-pooled to:

    14 × 14 × 384

---

## 3. Adaptive Hemispheric Difference Attention

Adaptive Hemispheric Difference Attention (AHDA) models interhemispheric feature differences.

The ConvNeXt Stage 3 representation and pooled Vision Mamba representation are both:

    14 × 14 × 384

They are concatenated to form:

    14 × 14 × 768

A 1 × 1 convolution followed by GELU projects the representation back to:

    14 × 14 × 384

The AHDA process is:

    14 × 14 × 384
           |
           v
    Central Sagittal Partition
           |
     +-----+-----+
     |           |
     v           v
Left Hemisphere  Right Hemisphere
14 × 7 × 384     14 × 7 × 384
                     |
                     v
            Horizontal Reflection
                     |
                     v
          Interhemispheric Comparison
                     |
                     v
             Absolute Difference
                     |
                     v
          Global Average Pooling
                     |
                     v
             Channel Descriptor
                     |
                     v
          Fully Connected Layers
                     |
                     v
                GELU + Sigmoid
                     |
                     v
          Adaptive Attention Weights
                     |
                     v
           Residual Recalibration
                     |
                     v
               14 × 14 × 384

AHDA uses the fixed image midpoint as the hemispheric reference.

The method does not perform:

- Per-image midline estimation
- Image registration
- Skull stripping

Therefore, AHDA represents a computational abstraction of bilateral cerebral anatomy and depends on approximately midline-centered framing.

---

## 4. Adaptive Cross-Attention Fusion

Adaptive Cross-Attention Fusion (ACAF) integrates the local ConvNeXt representation with the AHDA-enhanced representation.

Both representations have dimensions:

    14 × 14 × 384

They are flattened to:

    196 × 384

The local representation provides the Query, while the AHDA-enhanced representation provides the Key and Value.

    Local ConvNeXt Features
              |
              v
            Query
              |
              v
      Scaled Dot-Product
        Cross-Attention
              ^
              |
         Key + Value
              |
              ^
    AHDA-Enhanced Features
              |
              v
      Residual Aggregation
              |
              v
       Layer Normalization
              |
              v
          196 × 384
              |
              v
         14 × 14 × 384

The final fused representation is:

    14 × 14 × 384

---

## 5. Classification Head

The fused representation is passed through:

    14 × 14 × 384
            |
            v
    Global Average Pooling
            |
            v
          Dropout
            |
            v
    Fully Connected Layer
            |
            v
         Sigmoid
            |
            v
    Stroke Probability

The classification threshold is:

    0.5

where:

    1 → Stroke
    0 → Normal

---

## 6. Dataset

The study uses the publicly available Kaggle dataset:

**Brain Stroke Prediction CT Scan Image Dataset**

by Iashiqul.

The original dataset contains:

| Property | Description |
|---|---:|
| Original images | 2,515 |
| Normal | 1,551 |
| Stroke | 964 |
| Filename-defined case groups | 82 |
| Slices per case group | 19–40 |
| Median slices per case | 30 |
| Image resolution | 650 × 650 pixels |

Fourteen duplicate images were removed, resulting in:

    2,501 unique CT slice images

The images are rendered axial CT slices.

The dataset does not provide:

- Hounsfield units
- Display-window settings
- Acquisition parameters
- Slice ordering

The dataset also does not disclose acquisition protocol, inclusion criteria, diagnostic reference standard, annotation procedure, or stroke subtype.

The three-channel representation was preserved because the images contain identical grayscale intensities across the three channels.

### Dataset Source

https://www.kaggle.com/datasets/iashiqul/brain-stroke-prediction-ct-scan-image-dataset


---

## 7. Data Partitioning

All partitioning was performed at the **case-group level**.

Images sharing the same case identifier were assigned to the same subset.

| Subset | Case Groups | Images | Normal | Stroke |
|---|---:|---:|---:|---:|
| Training | 57 | 1,761 | 1,065 | 696 |
| Validation | 8 | 221 | 148 | 73 |
| Case-independent Test | 17 | 519 | 338 | 181 |
| **Total** | **82** | **2,501** | — | — |

No resampling or class-balancing procedures were applied.

Training and validation together form the development set:

    65 case groups
    1,982 images

The test set contains:

    17 case groups
    519 images

The test set was withheld from model development and comparative analysis and evaluated once using the final locked configuration.

### Important Dataset Qualification

The filename-defined case groups could not be cross-referenced with clinical records.

Therefore, correspondence between case groups and distinct patients could not be verified, and patient-level independence between subsets cannot be guaranteed.

---

## 8. Image Preprocessing

The original images are:

    650 × 650 × 3

and are resized to:

    224 × 224 × 3

The preprocessing pipeline is:

    Original CT Image
            |
            v
          Resize
      650 × 650 × 3
            |
            v
           CLAHE
      Clip Limit = 2.0
      Tile Grid = 8 × 8
            |
            v
    Gamma Correction
          γ = 1.2
            |
            v
      Normalization
          [0, 1]
            |
            v
      224 × 224 × 3

CLAHE is applied to one channel and the resulting image is duplicated across all three channels.

Gamma correction uses:

    γ = 1.2

---

## 9. Online Data Augmentation

Online augmentation is applied only to the training set.

| Augmentation | Range |
|---|---:|
| Random rotation | ±8° |
| Random zoom | ±15% |
| Random contrast | ±20% |

---

## 10. Adaptive Lévy Flight Gorilla Troops Optimizer

ALGTO extends Gorilla Troops Optimization with nonlinear parameter adaptation and stagnation-triggered Lévy-flight perturbation.

ALGTO optimizes:

- Learning rate
- Batch size
- Weight decay
- Dropout rate

The architecture remains fixed.

### Search Space

| Hyperparameter | Search Range |
|---|---|
| Learning rate | 0.00001–0.001 |
| Batch size | {8, 16, 32} |
| Weight decay | 0.00001–0.001 |
| Dropout | 0.10–0.50 |

### ALGTO Configuration

| Parameter | Value |
|---|---:|
| Population size | 8 |
| Maximum iterations | 12 |
| Maximum control coefficient | 1.0 |
| Nonlinear adaptation exponent | 2.0 |
| Stagnation threshold | 5 |
| Lévy exponent | 1.5 |
| Lévy scaling factor | 0.01 |
| Total model evaluations | 96 |

The fitness function is validation loss.

The candidate with the minimum validation loss is selected.

---

## 11. Final Training Configuration

The final configuration is:

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Epochs | 50 |
| Learning-rate scheduler | Cosine annealing |
| Learning rate | 0.00005 |
| Batch size | 16 |
| Weight decay | 0.0001 |
| Dropout | 0.30 |

The best validation-performing model state is retained.

---

## 12. Evaluation Protocol

The network produces slice-level Stroke probabilities.

For case-level evaluation, probabilities are averaged within each case group.

    Slice 1 Probability
            +
    Slice 2 Probability
            +
            ...
            +
    Slice N Probability
            |
            v
    Mean Stroke Probability
            |
            v
       Threshold = 0.5
            |
            v
      Case-Level Prediction

Each case group contributes equally to evaluation regardless of the number of slices.

Case-level evaluation includes:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Accuracy confidence intervals use the Wilson score method.

Paired baseline comparisons use McNemar's exact test with Holm–Bonferroni adjustment.

---

## 13. Locked Test Evaluation

The final configuration was determined using the training and validation subsets.

The case-independent test set was then evaluated once.

    Test Case Groups → 17
    Test Images      → 519

The test set was excluded from model development and comparative analysis.

The final configuration was subsequently locked.

---

## 14. Final Test Results

The proposed framework achieved:

| Metric | Result |
|---|---:|
| Accuracy | **94.12%** |
| Weighted F1-score | **94.20%** |
| ROC-AUC | **0.9697** |

These results were obtained on the locked, single-use internal case-level test set.

---

## 15. Five-Fold Cross-Validation

Case-grouped five-fold cross-validation was conducted exclusively on the development set.

    Development Set
    65 case groups
    1,982 images

Five non-overlapping folds were used.

    4 folds → Training
    1 fold  → Validation

The ALGTO-selected hyperparameter configuration was held fixed across all folds and was not re-optimized within each fold.

The resulting mean performance was:

| Metric | Mean ± SD |
|---|---:|
| Accuracy | **90.77 ± 3.44%** |
| Precision | **89.33 ± 9.83%** |
| Recall | **88.00 ± 10.95%** |
| F1-score | **87.92 ± 4.54%** |

---

## 16. Component Ablation Study

The component-wise ablation was performed on the development set using case-grouped five-fold cross-validation.

| Configuration | Accuracy (%) | Precision (%) | Recall (%) | F1-score (%) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| ConvNeXt-Tiny | 76.92 ± 5.44 | 71.33 ± 7.94 | 68.00 ± 10.95 | 69.21 ± 7.53 | 0.800 ± 0.079 |
| ConvNeXt-Tiny + Vision Mamba | 87.69 ± 4.21 | 84.67 ± 8.69 | 84.00 ± 8.94 | 83.96 ± 5.47 | 0.900 ± 0.050 |
| ConvNeXt-Tiny + Vision Mamba + AHDA | 89.23 ± 4.21 | 86.95 ± 12.68 | 88.00 ± 10.95 | 86.40 ± 4.56 | 0.925 ± 0.050 |
| ConvNeXt-Tiny + Vision Mamba + AHDA + ACAF | 89.23 ± 4.21 | 88.57 ± 15.65 | 88.00 ± 10.95 | 86.67 ± 3.04 | 0.950 ± 0.050 |

---

## 17. Optimization-Based Ablation

The complete architecture was evaluated using:

- Default hyperparameters
- Conventional GTO
- GTO with adaptive Lévy flight
- Proposed ALGTO

The proposed ALGTO achieved:

    Accuracy  → 90.77%
    Precision → 89.33%
    F1-score  → 87.92%
    ROC-AUC   → 0.950

The number of misclassified development case groups decreased from:

    7 → 6

with ALGTO.

---

## 18. Hyperparameter Sensitivity Analysis

Sensitivity analysis was performed using case-grouped five-fold cross-validation on the development set.

The selected configuration was:

    Learning rate → 0.00005
    Batch size    → 16
    Weight decay  → 0.0001
    Dropout       → 0.30

This configuration yielded:

    Accuracy → 90.77%
    F1-score → 87.92%

The sensitivity analysis showed a marked dependence of classification performance on the selected hyperparameters.

---

## 19. Convergence Analysis

ALGTO used:

    Population size     → 8
    Maximum iterations  → 12

resulting in:

    96 model evaluations

The best validation loss decreased rapidly during early evaluations and was followed by progressively smaller improvements.

---

## 20. Grad-CAM Visualization

Gradient-weighted Class Activation Mapping (Grad-CAM) was used for qualitative visualization.

Grad-CAM provides a qualitative illustration of the image regions influencing model predictions.

---

## 21. Implementation Environment

The experiments were performed using:

| Component | Version / Hardware |
|---|---|
| Platform | Google Colab Pro |
| GPU | NVIDIA Tesla T4 |
| Python | 3.12.13 |
| PyTorch | 2.11.0 |
| NumPy | 2.0.2 |
| CUDA | 13.0 |

The main pipeline used:

    Random seed = 42

The optimizer comparison was repeated across five random seeds.

---


**GitHub Repository**

https://github.com/fathimabeevi121/Hybrid-ConvNeXt-Tiny-Vision-Mamba-Stroke-Classification
