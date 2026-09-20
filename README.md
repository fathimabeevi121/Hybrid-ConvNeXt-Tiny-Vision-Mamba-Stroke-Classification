# Hybrid ConvNeXt-Tiny–Vision Mamba Framework for Brain Stroke Classification

This repository contains research notebooks and experimental implementations for a hybrid deep learning framework for binary brain stroke classification using axial computed tomography (CT) images.

The framework combines ConvNeXt-Tiny, Vision Mamba, Adaptive Hemispheric Difference Attention (AHDA), Adaptive Cross-Attention Fusion (ACAF), and Adaptive Lévy Flight Gorilla Troops Optimizer (ALGTO).

## Research Objectives

- Develop a hybrid ConvNeXt-Tiny–Vision Mamba architecture for CT-based stroke classification.
- Investigate hemispheric feature differences using AHDA.
- Integrate complementary feature representations using ACAF.
- Optimize training hyperparameters using ALGTO.
- Evaluate model components through ablation experiments and cross-validation.
- Support reproducible research through documented preprocessing and evaluation procedures.

## Dataset Details

The study uses the publicly available Brain Stroke Prediction CT Scan Image Dataset from Kaggle.

| Property | Description |
|---|---|
| Dataset | Brain Stroke Prediction CT Scan Image Dataset |
| Image modality | Brain computed tomography (CT) |
| Classification task | Binary classification |
| Classes | Normal and Stroke |
| Original images | 2,515 |
| Images after duplicate removal | 2,501 |
| Unique patient cases | 82 |
| Original image size | 650 × 650 pixels |
| Model input size | 224 × 224 × 3 |

### Dataset Source

[Brain Stroke Prediction CT Scan Image Dataset – Kaggle](https://www.kaggle.com/datasets/iashiqul/brain-stroke-prediction-ct-scan-image-dataset)

The dataset is not included in this repository. Users must download it from the original source and verify the applicable terms of use and licensing conditions.

## Patient-Level Data Splitting

Patient-level separation is used to reduce subject-level data leakage between training, validation, and test subsets.

| Subset | Patients | Images | Normal | Stroke |
|---|---:|---:|---:|---:|
| Training | 57 | 1,761 | 1,065 | 696 |
| Validation | 8 | 221 | 148 | 73 |
| Test | 17 | 519 | 338 | 181 |
| Total | 82 | 2,501 | 1,551 | 964 |

The patient-level split manifest should be retained with the experimental records to document case identifiers, class labels, and subset assignments.

## Preprocessing

The reported preprocessing pipeline includes:

1. Resize CT images from `650 × 650 × 3` to `224 × 224 × 3`.
2. Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
3. Apply gamma correction.
4. Normalize image values to the `[0, 1]` range.

### CLAHE Configuration

- Clip limit: `2.0`
- Tile grid size: `8 × 8`

### Training Augmentation

- Random rotation: `±8°`
- Random zoom: `±15%`
- Random contrast adjustment: `±20%`

Validation and test images are not augmented.

## Model Components

### ConvNeXt-Tiny

ConvNeXt-Tiny extracts hierarchical local spatial features from the CT images.

### Vision Mamba

Vision Mamba models long-range contextual dependencies in the extracted image representations.

### Adaptive Hemispheric Difference Attention (AHDA)

AHDA processes feature-level differences between corresponding left and right hemispheric regions.

The module uses a central sagittal reference and horizontal reflection of the contralateral feature representation.

AHDA is a computational feature-processing abstraction and is not intended to represent a physiological model of the brain.

### Adaptive Cross-Attention Fusion (ACAF)

ACAF integrates complementary local and stroke-enhanced feature representations before classification.

### Adaptive Lévy Flight Gorilla Troops Optimizer (ALGTO)

ALGTO is used for offline hyperparameter optimization.

The reported configuration includes nonlinear parameter adaptation, stagnation detection, and Lévy-flight perturbation.

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

## Experimental Notebooks

The repository includes notebooks for:

- Hybrid ConvNeXt-Tiny and Vision Mamba implementation
- ConvNeXt-Tiny ablation
- ConvNeXt-Tiny and Vision Mamba ablation
- AHDA ablation
- ACAF ablation
- K-fold cross-validation
- Conventional GTO
- Adaptive Lévy-flight GTO
- Model training without optimization
- Model comparison

The complete notebook filenames and execution requirements should be checked directly in the repository.

## Reproducibility

To reproduce the experiments:

1. Clone or download this repository.
2. Download the dataset from the original Kaggle source.
3. Arrange the dataset according to the paths expected by the notebooks.
4. Review all notebook configuration cells before execution.
5. Install the required Python libraries and compatible versions.
6. Confirm the image preprocessing configuration.
7. Use the documented patient-level split manifest.
8. Set and record the random seed used for each experiment.
9. Record the Python version, framework versions, GPU model, and operating system.
10. Run the preprocessing, training, validation, and evaluation notebooks.
11. Record the selected hyperparameters and model checkpoints.
12. Save generated figures, confusion matrices, and evaluation tables separately.
13. Ensure that test data is not used for model fitting or hyperparameter selection.

## Recommended Software Environment

The exact dependency versions should be confirmed from the notebooks and training environment.

Recommended documentation includes:

- Python version
- PyTorch version
- Torchvision version
- CUDA version, if applicable
- GPU model and memory
- Operating system
- Random seed
- Batch size
- Learning rate
- Number of training epochs
- Optimizer and scheduler settings
- Dataset split manifest

A `requirements.txt` file and a consolidated configuration file can be added after the final environment has been verified.

## Suggested Repository Structure

```text
Hybrid-ConvNeXt-Tiny-Vision-Mamba-Stroke-Classification/
│
├── README.md
├── notebooks/
├── data/
├── results/
│   ├── figures/
│   └── tables/
├── checkpoints/
├── requirements.txt
└── .gitignore
```

Patient data, private information, credentials, and restricted datasets must not be uploaded to the repository.

## Limitations

- The experiments use one publicly available dataset.
- External validation has not been performed.
- The task is limited to binary classification.
- The patient-level test cohort is relatively small.
- AHDA assumes approximately midline-centered CT framing.
- The dataset's acquisition and annotation procedures are not fully documented by the source repository.

## Ethical and Clinical Statement

This repository is intended for academic research and experimentation.

The models are not medical devices and must not be used as a substitute for professional clinical assessment, radiologist interpretation, or validated clinical decision-making.

## License

A source-code license has not yet been selected for this repository.

The dataset is subject to the terms and conditions of its original provider. Verify the dataset license before using or redistributing any data.
