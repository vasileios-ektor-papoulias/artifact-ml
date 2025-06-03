# Artifact-Torch Demo: Tabular Data Synthesis

This demo showcases how to use the artifact-torch framework to build, train, and evaluate a tabular data synthesizer with integrated validation from artifact-core and artifact-experiment.

## Overview

The demo implements a Variational Autoencoder (VAE) for tabular data synthesis and demonstrates how to:

1. Preprocess and load tabular data with mixed continuous and categorical features
2. Define a PyTorch model compatible with the artifact framework
3. Setup training and validation pipelines
4. Generate synthetic data
5. Evaluate the synthetic data quality using artifact-core's validation artifacts

## Components

The demo consists of the following files:

- **tabular_synthesis.ipynb**: Main notebook demonstrating the full workflow
- **model.py**: Implementation of the TabularSynthesizer model (VAE architecture)
- **dataset.py**: Data preprocessing and loading utilities
- **validation_plan.py**: Validation plan definition for synthetic data evaluation
- **validation_routine.py**: Validation routine that integrates with the training process
- **trainer.py**: Training components for the TabularSynthesizer

## Getting Started

To run the demo:

1. Make sure you have the full artifact-ml repository cloned
2. Install all dependencies for artifact-core, artifact-experiment, and artifact-torch
3. Open the `tabular_synthesis.ipynb` notebook in Jupyter
4. Run the cells in sequence to see the full workflow

## Dataset

The demo uses the Heart Disease dataset from artifact-core/assets/real.csv, which includes:
- Continuous features: Age, RestingBP, Cholesterol, MaxHR, Oldpeak
- Categorical features: Sex, ChestPainType, FastingBS, RestingECG, ExerciseAngina, ST_Slope, HeartDisease

## Model Architecture

The TabularSynthesizer implements a Variational Autoencoder (VAE) with:
- Encoder network to transform input data into a latent representation
- Mean and log-variance networks for sampling from the latent space
- Decoder network to generate synthetic data from the latent representation
- Loss function combining reconstruction loss and KL divergence

## Validation

The demo showcases artifact-core's validation capabilities:
- Distribution comparison (PDF plots, JS distance)
- Correlation structure analysis
- Descriptive statistics comparison
- Dimensionality reduction visualizations (PCA, t-SNE)

## Customization

This demo can be extended in several ways:
- Try different model architectures by modifying model.py
- Test with different datasets by changing the data loading and preprocessing
- Add additional validation metrics by extending the validation plan
- Experiment with different training parameters and regularization techniques 