# ⚙️ artifact-torch

> A deep learning framework built on top of artifact-core—abstracting engineering aspects of deep learning research to enable researchers to focus on what matters.


<p align="center">
  <img src="./assets/artifact_ml_logo.svg" width="400" alt="Artifact-ML Logo">
</p>


![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/github/license/vasileios-ektor-papoulias/artifact-core)

---

## 📋 Overview

This repository provides PyTorch integration for the **Artifact** framework.

It stands alongside:
- **artifact-core**: The foundation providing a flexible minimal interface for computing heterogeneous validation artifacts.
- **artifact-experiment**: Executable validation plans exporting results to popular experiment tracking services.

`artifact-torch` is a deep learning framework built on top of artifact-core—abstracting engineering aspects of deep learning research (data pipelines, device management, training, monitoring, validation) to enable researchers to focus on innovation.

## 🚀 Key Features

- **PyTorch Integration**: Lightweight PyTorch extension ensuring type-safety and compatibility with Artifact's validation plans
- **Extendible Trainer Abstraction**: Covers the full training/validation pipeline for thematically related neural networks
- **Model Management**: Standardized interfaces for model definition, saving, and loading
- **Data Pipeline Utilities**: Simplified data handling with type-safe interfaces
- **Validation Integration**: Seamless integration with artifact-core validation components

## 🏗️ Architecture and Components

### Trainer: The Backbone

The Trainer is the central architectural component of the framework, orchestrating the entire training process. It follows a clear hierarchical structure with the following responsibilities:

- **Training Loop Management**: Handles epoch iteration, batch processing, and stopping conditions
- **Device Management**: Ensures models and data are on the correct device
- **Optimization**: Coordinates optimizer and learning rate scheduler updates
- **State Management**: Maintains training state including model, optimizer, and metrics
- **Validation Integration**: Executes validation routines at configurable intervals

The trainer works with models that implement specific input/output interfaces, ensuring type safety and consistency across the framework. It utilizes data pipelines to feed the model and captures the results for validation and logging.

### Pluggable Component System

The framework's strength comes from its pluggable architecture, allowing components to be combined and configured for specific use cases:

1. **Models**: Implement the `Model` interface with standardized I/O through `ModelInput` and `ModelOutput` TypedDict interfaces
2. **Data Loaders**: Provide data to the trainer through the `DataLoader` interface
3. **Validation Routines**: Configure how and when validation is performed during training
4. **Callbacks**: Hook into the training process at specific points to add custom behaviors
5. **Stoppers**: Determine when training should terminate based on configurable conditions

These components can be mixed and matched to create custom training pipelines without modifying the core trainer logic.

### Component Details

#### Callback System

The callback system allows injecting custom behavior at various points in the training process:

- **Epoch Callbacks**: Execute at the beginning and end of each epoch
- **Batch Callbacks**: Execute before and after processing each batch
- **Validation Callbacks**: Execute during validation phases
- **Tracking Callbacks**: Integrate with experiment tracking platforms
- **Score, Array, and Plot Callbacks**: Generate and track metrics and visualizations

Callbacks can be combined and prioritized to create complex behaviors while keeping the trainer implementation clean.

#### Early Stopping System

The early stopping system provides mechanisms to terminate training based on various conditions:

- **Patience-Based Stopping**: Stops when metrics fail to improve for a specified number of epochs
- **Performance-Based Stopping**: Stops when metrics reach specified thresholds
- **Time-Based Stopping**: Limits training duration
- **Custom Stopping Criteria**: Allows implementing domain-specific stopping logic

The system is designed to be extensible, allowing new stopping conditions to be added without modifying existing code.

#### Model Tracking System

The model tracking system provides:

- **Metric Logging**: Records training and validation metrics
- **Artifact Storage**: Saves model checkpoints, visualizations, and other artifacts
- **Experiment Versioning**: Maintains history of training runs
- **Integration with External Tools**: Connects to tracking platforms like MLflow, Weights & Biases, etc.

This system ensures training progress is monitored and models are properly preserved for later use.

### Trainer Public Interface

The trainer's public interface provides methods for:

- **Training**: `train()` to initiate the training process
- **Validation**: `validate()` to manually trigger validation
- **Checkpoint Management**: `save_checkpoint()` and `load_checkpoint()` for model persistence
- **Metrics Access**: Methods to retrieve training and validation metrics

### Trainer Subclass Hooks

Trainers can be customized by implementing the following hooks in subclasses:

- **`_should_stop()`**: Determines when training should terminate
- **`_execute_batch_preprocessing()`**: Prepares each batch before model processing
- **`_execute_batch_postprocessing()`**: Handles model outputs after each batch
- **`_execute_epoch_preprocessing()`**: Prepares for each training epoch
- **`_execute_epoch_postprocessing()`**: Finalizes each training epoch
- **`_log_batch_metrics()`**: Defines what metrics to log from each batch
- **`_log_epoch_metrics()`**: Defines what metrics to log from each epoch

### Validation Routine Abstraction

The validation routine provides a standardized way to evaluate models during training:

- **Validation Scheduling**: Controls when validation occurs during training
- **Validation Logic**: Defines how the model is evaluated
- **Artifact Generation**: Creates validation artifacts using artifact-core components
- **Metric Tracking**: Records validation metrics for monitoring and early stopping

### Validation Routine Subclass Hooks

Validation routines can be customized by implementing:

- **`_get_validation_period()`**: Determines how often validation runs
- **`_get_validation_plan()`**: Specifies which validation components to use
- **`_get_train_loader_callbacks()`**: Configures callbacks for training data
- **`_get_val_loader_callbacks()`**: Configures callbacks for validation data
- **`_execute_validation()`**: Implements the validation logic

### Core Folder Organization

The `core/` folder contains common logic shared across different implementations:

- **Base Classes**: Abstract classes defining interfaces
- **Utility Functions**: Common operations used throughout the framework
- **Shared Components**: Components that apply to multiple domains

This organization promotes code reuse and ensures consistent behavior across different implementations.

### Artifact Group Organization

Each artifact group from artifact-core has its own directory in the root of this project:

- **table_comparison/**: Components for tabular data synthesis and comparison
- **image_comparison/**: Components for image generation and comparison (future)
- **text_comparison/**: Components for text generation and comparison (future)

#### Table Comparison Module

The table comparison module provides specialized components for tabular data:

- **TabularGenerativeModel**: Base class for models that generate tabular data
- **TableComparisonValidationRoutine**: Validation routine for comparing real and synthetic tabular data
- **TableComparisonTrainer**: Specialized trainer for tabular data synthesis models

These components integrate with artifact-core's table comparison validation plans to enable comprehensive evaluation of synthetic tabular data.

## 🚀 Building Projects with artifact-torch

To build a project based on artifact-torch, you need to develop the following components:

1. **Model Implementation**: Create a model that implements the appropriate interfaces for your task
2. **Data Pipeline**: Implement data loading and preprocessing for your specific data
3. **Validation Configuration**: Configure validation routines specific to your task
4. **Training Configuration**: Set up a trainer with appropriate parameters and callbacks

The demo in this repository shows a complete implementation of a tabular data synthesizer, including:

- A VAE-based model implementing the TabularGenerativeModel interface
- Data preprocessing for mixed continuous and categorical features
- A validation routine with metrics specific to tabular data synthesis
- A trainer configured for the tabular synthesis task

By following this pattern and leveraging the existing components, you can quickly develop sophisticated deep learning systems with built-in validation capabilities.

## 🔧 Extending the Framework

### 1. Adding New Component Implementations

To add new implementations of existing components, place them in the appropriate `libs/components` directory:

1. Identify the component type you want to extend (e.g., callback, stopper, model)
2. Create a new class that inherits from the appropriate base class
3. Implement the required methods and interfaces
4. Register the component if required by the component system

For example, to add a new callback:

1. Identify the callback type (epoch, batch, validation)
2. Create a class that inherits from the appropriate callback base class
3. Implement the required callback methods

### 2. Adding New Artifact Groups

To add support for a new artifact group (similar to table_comparison):

1. Create a new directory at the root level named after the artifact group
2. Define the base interfaces for the new domain
3. Implement model interfaces specific to the domain
4. Create specialized trainers and validation routines
5. Implement data handling components for the domain
6. Ensure integration with corresponding artifact-core validation components

This structure allows the framework to be extended to new domains while maintaining consistency with the existing architecture.

## 🚀 Installation

### Using Poetry (Recommended)

```bash
git clone https://github.com/vasileios-ektor-papoulias/artifact-torch.git
cd artifact-torch
poetry install
```

### Using Pip

```bash
git clone https://github.com/vasileios-ektor-papoulias/artifact-torch.git
cd artifact-torch
pip install .
```

## 🤝 Contributing

Contributions are welcome! 

Please feel free to submit a Pull Request following the guidelines in [the general Artifact-ML README](https://github.com/vasileios-ektor-papoulias/artifact-ml/blob/main/README.md).

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
