# Motivating Example

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>  

## Table Synthesizer Training Workflow
We present the *same* tabular synthesis experiment expressed in two styles:

- an imperative script tightly coupled to a specific model, 
- a declarative workflow powered by Artifact-ML.

We assume the following are available in the execution context:

- **`real_data`** — a `pandas.DataFrame` containing the original (ground-truth) tabular dataset.
- **`config`** — a `dict` holding all required experiment configuration metadata.

### Imperative Experiment Script

First, we present a schematic imperative implementation of the workflow.

Even though high-level intent applies to any tabular synthesis experiment, the script is tightly coupled to a specific model.

Consequently, it is verbose and riddled with impertive glue code.

# Assume we have access to:
# a pandas dataframe named "real_data", "synthetic_data" and a config dict name "config"

```python
from typing import List, Dict, Tuple, Optional, Any
import sklearn.metrics
import sklearn.decomposition
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mlflow
import os
from scipy import stats
import torch
from torch.utils.data import Dataset, DataLoader

dataset = Dataset(real_data)
data_loader = DataLoader(
    dataset=dataset,
    batch_size=config.batch_size
    )
model = MyModel(**config)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.lr
    )
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config.step_size
    )

# Manual training loop with integrated validation
for epoch in range(config.num_epochs):
    model.train()
    epoch_loss = 0
    for batch_idx, batch in enumerate(data_loader):
    # Training step
        optimizer.zero_grad()
        # This model expects batch as dict with 'features' and 'targets' keys
        batch_dict = {'features': batch[:, :-1], 'targets': batch[:, -1:]}
        # This model includes loss in forward pass output - signature mismatch!
        outputs, loss = model.forward_training(batch_dict)  # Returns (outputs, loss) tuple
        # Other models might return just outputs, requiring separate loss computation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()
    
    # Periodic validation
    if epoch % config.validation_frequency == 0:
        model.eval()  # Switch to eval mode
        print(f"Running validation at epoch {epoch}...")
        
        # Compute validation loss using data loader
        with torch.no_grad():
            avg_loss = compute_loss(
                model=model,
                data_loader=data_loader
                )

        # Generate synthetic data for validation
        synthetic_data, metadata = generate_synthetic_data(
            model,
            n_samples=config.n_samples, # Model-specific generation hyperparameters
            temperature=config.temperature, # might differ for competing models
        )
        
        # Compute validation artifacts comparing real vs synthetic data
        mean_errors = compute_mean_errors(
            real_data=real_data,
            synthetic_data=synthetic_data,
        )
        corr_distance = compute_correlation_distance(
            real_data=real_data,
            synthetic_data=synthetic_data,
        )
        dist_fig = create_distribution_plots(
                real_data=real_data,
                synthetic_data=synthetic_data
                )

        # Some artifacts require extra hyperparameters
        categorical_columns = extract_categorical_columns(
            synthetic_data=synthetic_data,
            metadata=metadata
            )
        js_distances = compute_js_distance_per_column(
            real_data=real_data,
            synthetic_data=synthetic_data,
            categorical_columns=categorical_columns
            )

        
        # Log validation metrics
        mlflow.log_metric("avg_loss", avg_loss, step=epoch)
        mlflow.log_metric("mean_absolute_error", mean_errors.mean(), step=epoch)
        mlflow.log_metric("mean_js_distance", np.mean(js_distances), step=epoch)
        mlflow.log_metric("correlation_distance", corr_distance, step=epoch)

        # Log plots
        dist_filename = f"distributions_epoch_{epoch}.png"
        dist_fig.savefig(
            dist_filename,
            dpi=150,
            bbox_inches='tight'
            )
        mlflow.log_artifact(dist_filename)
            
        # Cleanup
        os.remove(dist_filename)
        plt.close(dist_fig)
        
        # Print progress
        print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, ValLoss={avg_val_loss:.4f}")
        
        # Switch back to training mode
        model.train()

print("Training completed!")
```
The model utilizes the following validation utils, tuned to model implementation via imperative adapter code.

```python
from collections import Counter

def compute_loss(
    model: Any,
    data_loader: Any
    ) -> float:
    val_loss = 0.0
    val_batches = 0
    for val_batch in data_loader:
        with torch.no_grad():
            # Imperative adaptation to model profile
            val_batch_dict = {
                'features': val_batch[:, :-1],
                'targets': val_batch[:, -1:]
                }
            val_outputs, batch_val_loss = model.forward_training(val_batch_dict)
            # Different models might need: val_loss = criterion(model.forward(val_batch), targets)
            val_loss += batch_val_loss.item()
            val_batches += 1
    return val_loss / val_batches if val_batches > 0 else float('inf')

def generate_synthetic_data(
    model: Any,
    n_samples: int = 1000,
    temperature: float = 0.8,
    top_k: int = 50
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # This model's generate method returns (synthetic_data, metadata) tuple
    synthetic_data, metadata = model.generate_samples(
        num_samples=n_samples,   # This model uses 'num_samples' not 'n_samples'
        temp=temperature,        # This model uses 'temp' not 'temperature'
        return_metadata=True
    )
    return synthetic_data, metadata

def extract_categorical_columns(
    data: pd.DataFrame,
    metadata: Dict[str, Any]
    ) -> List[str]:
    # The model in question returns static metadata alongside the synthetic data it generates.
    # These happen to include categorical column indices, rather than names.
    categorical_indices = metadata.get('categorical_cols', [])
    categorical_columns = [data.columns[i] for i in categorical_indices]
    return categorical_columns

def compute_mean_errors(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame
) -> np.ndarray:
    mean_errors = abs(real_data.mean() - synthetic_data.mean())
    return mean_errors

def compute_correlation_distance(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame
    ) -> float:
    # Extract numeric columns - correlation can't handle mixed types
    real_numeric = real_data.select_dtypes(include=[np.number])
    synthetic_numeric = synthetic_data.select_dtypes(include=[np.number])
    real_corr = np.corrcoef(real_numeric.T)
    synthetic_corr = np.corrcoef(synthetic_numeric.T)
    return np.linalg.norm(real_corr - synthetic_corr)

def compute_js_distance_per_column(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    categorical_columns: List[str]
    ) -> List[float]:
    # JS distance requires different handling for categorical vs continuous cols
    js_distances = []
    for col in real_data.columns:
        if col in categorical_columns:
            # Categorical: can compute JS distance directly
            js_dist = _js_distance_categorical(real_data[col], synthetic_data[col])
        else:
            # Continuous: JS distance calculation requires binning first
            js_dist = _js_distance_continuous_binned(
                real_data[col], synthetic_data[col], 
                bins=20  # Binning parameter not provided by model, must specify
            )
        js_distances.append(js_dist)
    return js_distances

def _js_distance_categorical(
    real_col: pd.Series,
    synthetic_col: pd.Series
    ) -> float:
    real_counts = Counter(real_col)
    synthetic_counts = Counter(synthetic_col)
    all_categories = set(real_counts.keys()) | set(synthetic_counts.keys())
    real_probs = np.array([real_counts.get(cat, 0) for cat in all_categories])
    synthetic_probs = np.array([synthetic_counts.get(cat, 0) for cat in all_categories])
    real_probs = real_probs / real_probs.sum() + 1e-8
    synthetic_probs = synthetic_probs / synthetic_probs.sum() + 1e-8
    m = 0.5 * (real_probs + synthetic_probs)
    return 0.5 * stats.entropy(real_probs, m) + 0.5 * stats.entropy(synthetic_probs, m)

def _js_distance_continuous_binned(
    real_col: pd.Series,
    synthetic_col: pd.Series,
    bins: int
    ) -> float:
    min_val = min(real_col.min(), synthetic_col.min())
    max_val = max(real_col.max(), synthetic_col.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    real_binned = np.digitize(real_col, bin_edges)
    synthetic_binned = np.digitize(synthetic_col, bin_edges)
    return _js_distance_categorical(real_binned, synthetic_binned)

def create_distribution_plots(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, plot_columns: Optional[List[str]] = None) -> plt.Figure:
    if plot_columns is None:
        plot_columns = real_data.select_dtypes(include=[np.number]).columns
    fig, axes = plt.subplots(len(plot_columns), 2, figsize=(15, 4 * len(plot_columns)))
    if len(plot_columns) == 1:
        axes = axes.reshape(1, -1)
    for i, col in enumerate(plot_columns):
        axes[i, 0].hist(real_data[col].dropna(), alpha=0.7, bins=30)
        axes[i, 0].set_title(f'{col} - Real')
        axes[i, 1].hist(synthetic_data[col].dropna(), alpha=0.7, bins=30)
        axes[i, 1].set_title(f'{col} - Synthetic')
    plt.tight_layout()
    return fig
```

### Reusable Experiment Workflow Built Declaratively

We now present a schematic implementation of the same workflow built with Artifact-ML.

The end result is **reusable** by any compatible model.

Every line of code **declares intent**, resulting in a compact and expressive workflow with significantly less code overhead.

**Model Implementation** - Isolated architecture design:

```python
class MyModel(TableSynthesizer[ModelInput, ModelOutput, GenerationParams]): # Generic IO profile and generation hyperparams
    def forward(self, batch: ModelInput) -> ModelOutput: ...
    
    def generate(self, generation_params: GenerationParams) -> pd.DataFrame: ...
```

**Validation Plan** - Specification of desired validation artifacts built declaratively (via subclass hooks):

```python
class MyValidationPlan(TableComparisonPlan):
    @staticmethod
    def _get_score_types() -> List[ScoreType]:
        return [
          ScoreType.MEAN_JS_DISTANCE,
          ScoreType.CORRELATION_DISTANCE,
          ]
    
    @staticmethod
    def _get_plot_types() -> List[PlotType]:
        return [
          PlotType.PDF
          ]
    
    @staticmethod
    def _get_array_collection_types() -> List[ArrayCollectionType]:
        return [
          ArrayCollectionType.MEAN_JUXTAPOSITION
          ]
```

**Artifact Validation Routine** - Reusable validation plan executor built declaratively (via subclass hooks):

```python
class MyArtifactRoutine(TableComparisonRoutine[GenerationParams]):
    @staticmethod
    def _get_validation_plan() -> TableComparisonPlan:
        return MyValidationPlan()
    
    @staticmethod
    def _get_generation_params() -> GenerationParams:
        return GenerationParams(
            num_samples=config.num_samples,
            temperature=config.temperature
            )
```

**Data Loader Routine** - Reusable callback executor built declaratively (via subclass hooks):

```python
class MyDataLoaderRoutine(
    DataLoaderRoutine[ModelInput, ModelOutput] # Compatible expected IO profile
    ):
    @staticmethod
    def _get_score_callbacks() -> List[
        DataLoaderScoreCallback[ModelInput, ModelOutput]
        ]:
        return [
            TrainLossCallback()
            ]
```

**Trainer Configuration** - Reusable training loop built declaratively (via subclass hooks):

```python
class MyTrainer(
    CustomTrainer[
        TableSynthesizer[ModelInput, ModelOutput, Any], # Trainer works with any tabular synthesizer adhering to expected IO profile.
        ModelInput, # Compatible expected forward pass input.
        ModelOutput, # Compatible expected forward pass output.
        ModelTrackingCriterion, # Trainer works with base model tracking criterion (model tracking unused).
        StopperUpdateData # Trainer works with base stopper update data (used simple epoch bound stopper).
    ]
):
    def _get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr
            )
    
    def _get_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.step_size
            )
    
    def _get_early_stopper(self) -> EarlyStopper:
        return EpochBoundStopper(
            n_epochs=config.num_epochs
            )
    
    @staticmethod
    def _get_train_loader_routine(
        data_loader: DataLoader[ModelInputT], # Typed Artifact-ML wrapper of native torch DataLoader
        tracking_client: Optional[TrackingClient], # Artifact-ML experiment tracking client (for logging)
    ) -> Optional[DataLoaderRoutine[ModelInputT, ModelOutputT]]:
        return DemoLoaderRoutine.build(
            data_loader=data_loader,
            tracking_client=tracking_client
            )
```

**Experiment Execution** - Complete training, validation, and experiment tracking in just a few lines:

```python
data_spec = DataSpec(*config) # Static information about the real data e.g. feature names

dataset = Dataset(real_data)

data_loader = DataLoader(
    dataset=dataset,
    batch_size=config.batch_size
    )

model = MyModel(*config)

artifact_routine = MyArtifactRoutine.build(
        data=real_data,
        data_spec=data_spec,
        tracking_client=tracking_client
        )

tracking_client = TrackingClient(*config)

trainer = MyTrainer.build(
    model=model,
    train_data_loader=data_loader,
    artifact_routine=artifact_routine,
    tracking_client=tracking_client
)

results = trainer.train()  # Training + validation + tracking automatically handled
```