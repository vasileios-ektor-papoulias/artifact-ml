# Motivating Example

This document provides a practical schematic example showcasing the problem addressed by Artifact-ML.

| ![Artifact-ML Logo](assets/artifact_ml_logo.svg){ width="400" } |
|:--:|

## 🔧 Ad Hoc vs. Systematic Experiment Workflows

The following exhibits the same tabular synthesis experiment implemented two ways:

### Ad-Hoc Experiment Script

First, we present a schematic ad-hoc implementation of the workflow.

Even though the hgh-level intent could apply to any tabular synthesis experiment, the script is designed to train a specific model.

Consequently, it is riddled with impertive glue code.

```python
# Monolithic experiment script demonstrating fundamental problems despite intent to keep code organized
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

# ===== VALIDATION UTILITY FUNCTIONS =====
# Each validation metric has specific requirements forcing custom adapter code

def generate_synthetic_data(model: Any, n_samples: int = 1000, temperature: float = 0.8, top_k: int = 50) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate data with model-specific generation parameters"""
    # This model's generate method returns (synthetic_data, metadata) tuple
    synthetic_data, metadata = model.generate_samples(
        num_samples=n_samples,   # This model uses 'num_samples' not 'n_samples'
        temp=temperature,        # This model uses 'temp' not 'temperature'
        top_k=top_k,
        return_metadata=True
    )
    return synthetic_data, metadata

def extract_categorical_columns(data: pd.DataFrame, metadata: Dict[str, Any]) -> List[str]:
    """Extract categorical column info for JS distance calculation"""
    # This model stores categorical column indices, not names
    categorical_indices = metadata.get('categorical_cols', [])
    categorical_columns = [data.columns[i] for i in categorical_indices]
    return categorical_columns

def compute_js_distance_per_column(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, categorical_columns: List[str]) -> List[float]:
    """JS distance requires different handling for categorical vs continuous"""
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

def _js_distance_categorical(real_col: pd.Series, synthetic_col: pd.Series) -> float:
    """JS distance for categorical data"""
    from collections import Counter
    real_counts = Counter(real_col)
    synthetic_counts = Counter(synthetic_col)
    
    all_categories = set(real_counts.keys()) | set(synthetic_counts.keys())
    real_probs = np.array([real_counts.get(cat, 0) for cat in all_categories])
    synthetic_probs = np.array([synthetic_counts.get(cat, 0) for cat in all_categories])
    
    real_probs = real_probs / real_probs.sum() + 1e-8
    synthetic_probs = synthetic_probs / synthetic_probs.sum() + 1e-8
    
    m = 0.5 * (real_probs + synthetic_probs)
    return 0.5 * stats.entropy(real_probs, m) + 0.5 * stats.entropy(synthetic_probs, m)

def _js_distance_continuous_binned(real_col: pd.Series, synthetic_col: pd.Series, bins: int) -> float:
    """JS distance for continuous data requires binning"""
    min_val = min(real_col.min(), synthetic_col.min())
    max_val = max(real_col.max(), synthetic_col.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    real_binned = np.digitize(real_col, bin_edges)
    synthetic_binned = np.digitize(synthetic_col, bin_edges)
    
    return _js_distance_categorical(real_binned, synthetic_binned)

def compute_correlation_distance(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> float:
    """Correlation distance calculation requires numeric data only"""
    # Extract numeric columns - correlation can't handle mixed types
    real_numeric = real_data.select_dtypes(include=[np.number])
    synthetic_numeric = synthetic_data.select_dtypes(include=[np.number])
    
    real_corr = np.corrcoef(real_numeric.T)
    synthetic_corr = np.corrcoef(synthetic_numeric.T)
    
    return np.linalg.norm(real_corr - synthetic_corr)

def compute_pca_projection(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """PCA projection requires specific data preprocessing"""
    # PCA needs numeric data with no NaN values
    real_numeric = real_data.select_dtypes(include=[np.number]).fillna(0)
    synthetic_numeric = synthetic_data.select_dtypes(include=[np.number]).fillna(0)
    
    pca = sklearn.decomposition.PCA(n_components=n_components)
    real_pca = pca.fit_transform(real_numeric)
    synthetic_pca = pca.transform(synthetic_numeric)
    
    return real_pca, synthetic_pca

def compute_unique_value_counts(data: pd.DataFrame) -> Dict[str, int]:
    """Some validation metrics need derived statistics, not raw DataFrames"""
    # This metric requires unique value counts per column, not the DataFrame itself
    unique_counts = {}
    for col in data.columns:
        unique_counts[col] = data[col].nunique()
    return unique_counts

def compute_validation_loss(model: Any, val_data_loader: Any) -> float:
    """Compute validation loss on validation data loader"""
    val_loss = 0.0
    val_batches = 0
    for val_batch in val_data_loader:
        with torch.no_grad():
            # Same signature mismatch issue applies to validation!
            val_batch_dict = {'features': val_batch[:, :-1], 'targets': val_batch[:, -1:]}
            val_outputs, batch_val_loss = model.forward_training(val_batch_dict)
            # Different models might need: val_loss = criterion(model.forward(val_batch), targets)
            val_loss += batch_val_loss.item()
            val_batches += 1
    return val_loss / val_batches if val_batches > 0 else float('inf')

def create_distribution_plots(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, plot_columns: Optional[List[str]] = None) -> plt.Figure:
    """Distribution plotting function with configurable parameters"""
    if plot_columns is None:
        # Plotting defaults to numeric columns only
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

# ===== MAIN EXPERIMENT LOGIC =====

# Manual training loop with integrated validation
model = MyCustomModel(hidden_dim=128, num_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)

num_epochs = 100
batch_size = 64
validation_frequency = 10  # Run validation every 10 epochs

for epoch in range(num_epochs):
    # Training phase
    model.train()
    epoch_loss = 0
    
    for batch_idx, batch in enumerate(train_data_loader):
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
    
    # Periodic validation phase
    if epoch % validation_frequency == 0:
        model.eval()  # Switch to eval mode
        
        print(f"Running validation at epoch {epoch}...")
        
        # Generate synthetic data with model-specific parameters
        with torch.no_grad():
            synthetic_data, metadata = generate_synthetic_data(
                model, n_samples=1000, temperature=0.8, top_k=50
            )
        
        # Extract categorical columns for JS distance calculation
        categorical_columns = extract_categorical_columns(synthetic_data, metadata)
        
        # Compute validation loss using data loader
        avg_val_loss = compute_validation_loss(model, val_data_loader)
        
        # Compute validation metrics comparing real vs synthetic data
        mean_errors = abs(real_data.mean() - synthetic_data.mean())
        js_distances = compute_js_distance_per_column(real_data, synthetic_data, categorical_columns)
        corr_distance = compute_correlation_distance(real_data, synthetic_data)
        real_pca, synthetic_pca = compute_pca_projection(real_data, synthetic_data)
        
        # Some metrics need derived data, not raw DataFrames
        real_unique_counts = compute_unique_value_counts(real_data)
        synthetic_unique_counts = compute_unique_value_counts(synthetic_data)
        unique_count_diff = {col: abs(real_unique_counts[col] - synthetic_unique_counts[col]) 
                            for col in real_unique_counts}
        
        # Log validation metrics (with epoch suffix to track progress)
        mlflow.log_metric("validation_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("mean_absolute_error", mean_errors.mean(), step=epoch)
        mlflow.log_metric("mean_js_distance", np.mean(js_distances), step=epoch)
        mlflow.log_metric("correlation_distance", corr_distance, step=epoch)
        mlflow.log_metric("mean_unique_count_diff", np.mean(list(unique_count_diff.values())), step=epoch)
        
        # Create and save plots periodically (every 20 epochs to avoid spam)
        if epoch % 20 == 0:
            dist_fig = create_distribution_plots(real_data, synthetic_data)
            pca_fig, pca_ax = plt.subplots(figsize=(8, 6))
            pca_ax.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='Real')
            pca_ax.scatter(synthetic_pca[:, 0], synthetic_pca[:, 1], alpha=0.5, label='Synthetic')
            pca_ax.legend()
            
            # Save with epoch number to avoid overwriting
            dist_filename = f"distributions_epoch_{epoch}.png"
            pca_filename = f"pca_epoch_{epoch}.png"
            
            dist_fig.savefig(dist_filename, dpi=150, bbox_inches='tight')
            pca_fig.savefig(pca_filename, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(dist_filename)
            mlflow.log_artifact(pca_filename)
            
            # Cleanup
            os.remove(dist_filename)
            os.remove(pca_filename)
            plt.close(dist_fig)
            plt.close(pca_fig)
        
        print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, ValLoss={avg_val_loss:.4f}, "
              f"MAE={mean_errors.mean():.4f}, JS={np.mean(js_distances):.4f}, Corr={corr_distance:.4f}")
        
        # Switch back to training mode
        model.train()

print("Training completed!")
```
**Key Problems:**

- **Repetitive preparation obscures intent**: Researchers solve the same preparation problem repeatedly to accommodate context-specific requirements, writing extensive imperative code unrelated to their intent---declaring "compute PCA projection" or "generate marginal plots." This mismatch between declarative goals and imperative implementation creates pervasive duplication, where identical preparation logic is rewritten countless times, suffering all the maintenance penalties despite expressing conceptually identical requirements.
- **Extensive duplication for minor differences**: Adapting validation logic to different models necessitates duplicating entire experiment scripts (200+ lines) to accommodate minor interface differences in batch formatting, method signatures (e.g. `forward` io types), and parameter naming. Even well-factored utility functions like `compute_validation_loss()` require model-specific adaptations, resulting in substantial redundancy for trivial disparities.
- **Brittle maintenance from tight coupling**: Modifying validation behavior—whether introducing new metrics or changing experiment tracking backends—requires editing monolithic scripts where changes propagate through coupled validation logic, increasing error risk.

**Consequences for Research Productivity:**

- **Validation changes require synchronized updates**: Each model maintains its own nearly-identical experiment script with model-specific adapter code. Changing validation logic requires implementing the same modification across all scripts while carefully adapting each change to individual interfaces.
- **Model evolution forces script maintenance**: Model modifications—even when validation logic remains unchanged—necessitate updating the corresponding experiment scripts to accommodate interface changes. Research teams expend substantial effort maintaining experiment infrastructure rather than advancing innovation.
- **Unintended differences create biased comparisons**: When maintaining multiple brittle validation pipelines, subtle discrepancies can emerge---different preprocessing steps, inconsistent hyperparameters, or varying evaluation metrics. This can lead to unreliable model comparisons where performance differences reflect implementation accidents rather than true model capabilities.

### Systematic Experiment Script

We now present a schematic systematic implementation of the same workflow.

The same workflow could work for any tabular synthesizer.

Every line of code declares intent, consequently the script is considerably shorter.


**1. Model Implementation** - Researchers---adhering to lightweight interface contracts---focus soley on architectural innovation:

```python
class MyModel(TableSynthesizer):
    def forward(self, batch: torch.Tensor) -> torch.Tensor: ...
    
    def generate(self, generation_params: GenerationParams) -> torch.Tensor: ...
```

**2. Validation Plan** - Declarative specification of desired validation artifacts:

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
          PlotType.PDF,
          PlotType.PCA_JUXTAPOSITION,
          ]
    
    @staticmethod
    def _get_array_collection_types() -> List[ArrayCollectionType]:
        return [
          ArrayCollectionType.MEAN_JUXTAPOSITION,
          ArrayCollectionType.STD_JUXTAPOSITION
          ]
```

**3. Artifact Validation Routine** - Domain-specific orchestrator that manages validation resource acquisition and artifact validation plan execution

```python
class MyArtifactRoutine(TableComparisonRoutine):
    @staticmethod
    def _get_validation_plan() -> TableComparisonPlan:
        return MyValidationPlan()
    
    @staticmethod
    def _get_generation_params() -> GenerationParams:
        return GenerationParams(special_param=True, num_samples=1000)
```

**4. Data Loader Routine** - Specifies validation callbacks for data loader-based metrics:

```python
class MyValLoaderRoutine(DataLoaderRoutine):
    @staticmethod
    def _get_score_callbacks() -> List[DataLoaderScoreCallback]:
        return [ValLossCallback()]
```

**5. Trainer Configuration** - Training loop configuration via subclass hooks:

```python
class MyTrainer(CustomTrainer):
    def _get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def _get_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30)
    
    def _get_early_stopper(self) -> EarlyStopper:
        return EarlyStopper(patience=10, min_delta=0.001)
```

**6. Experiment Execution** - Complete training, validation, and experiment tracking in just a few lines:

```python
artifact_routine = MyArtifactRoutine.build(
        data=val_data,
        data_spec=val_data_spec,
        tracking_client=tracking_client
        )

val_loader_routine = MyValLoaderRoutine.build(
        val_data_loader=val_data_loader,
        tracking_client=tracking_client
        )

trainer = MyTrainer.build(
    model=MyModel(architecture_config),
    artifact_routine=artifact_validation_routine,
    val_loader_routine=val_loader_routine,
    tracking_client=tracking_client
)

results = trainer.train()  # Training + validation + tracking automatically handled
```

**Key Benefits:**

- **Declarative workflows**: Researchers can express validation intent directly ("compute PCA projection," "generate marginal plots") without writing repetitive preparation code, eliminating the mismatch between declarative goals and imperative implementation.
- **Elimination of code duplication**: Minor interface variations across competing models no longer necessitate the duplication of entire experiment scripts.
- **Maintainable, decoupled logic**: Modular validation modifications no longer propagate changes through coupled monolithic scripts, reducing maintenance overhead.
- **Reusable infrastructure**: The experiment infrastructure developed (validation plan and training loop) will work with ANY tabular synthesizer. Researchers can import and use it directly, enabling full focus on innovation: they might not need to produce any experiment script at all to begin with. 