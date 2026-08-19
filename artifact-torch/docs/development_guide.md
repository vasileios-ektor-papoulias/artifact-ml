# Development Guide

<p align="center">
  <img src="../assets/artifact_ml_logo.svg" width="200" alt="Artifact-ML Logo">
</p>

## Package Layout

The `artifact_torch` package separates private framework internals from the public API:

- `_base/`: framework core—model contracts, trainer, experiment, data primitives, routine/plan/callback bases, and infrastructure components (caching, early stopping, model tracking).
- `_impl/`: concrete implementations of framework components (e.g. ready-made callbacks and early stoppers).
- `_domains/`: shared domain contracts reused across domain toolkits (e.g. `generation/` with `GenerativeModel`/`GenerationParams`, `classification/` with `Classifier`/`ClassificationParams`).
- `nn/`: the public facade for domain-agnostic components (`Model`, `ModelInput`, `ModelOutput`, `Dataset`, `DataLoader`, `Trainer`, plus the `routines`, `plans`, `callbacks`, `early_stopping` and `model_tracking` submodules).
- `spi/`: the service-provider interface for extension authors (`Experiment`, `ArtifactRoutine`, `ArtifactRoutineData`, `ArtifactRoutineHyperparams`).
- `table_comparison/`, `binary_classification/`: domain toolkits exposing domain models, artifact routines and experiments.

## Adding Domain Toolkits

Domain toolkits are packages under `artifact_torch/` (see `table_comparison/` and `binary_classification/` for reference). To add one:

1. **Domain Package**: Create `artifact_torch/domain_name/` with private modules (`_model.py`, `_routine.py`, `_experiment.py`) and a public `__init__.py` re-exporting the toolkit surface (including relevant `artifact-core` artifact types and the `artifact-experiment` plan for the domain).
2. **Shared Contracts**: If the domain introduces a reusable model capability (e.g. generation, classification), place the generic contract in `_domains/` and specialize it in the domain package (e.g. `TableSynthesizer` specializes `GenerativeModel` from `_domains/generation/`).
3. **Artifact Routine**: Subclass `ArtifactRoutine` (from `_base/components/routines/artifact.py`, exposed via `artifact_torch.spi`). Define the domain's routine data (an `ArtifactRoutineData` dataclass) and hyperparams (an `ArtifactRoutineHyperparams` dataclass), implement `_generate_artifact_resources` (e.g. call `model.generate(...)` and pair the output with the real data), and leave `_get_period(data_split)` / `_get_artifact_plan(data_split)` abstract for users. Routines are built with `DataSplit`-keyed data via `.build(data=..., data_spec=..., tracking_queue=...)`.
4. **Domain Experiment**: Subclass `Experiment` (from `_base/experiment/experiment.py`, exposed via `artifact_torch.spi`), pinning the domain's routine data and resource spec types. Users then implement `_get_trainer`, `_get_train_diagnostics_routine`, `_get_loader_routine` and `_get_artifact_routine`.

## Component Extension

**Model Type Contract Development**: Define new `Model`, `ModelInput` and `ModelOutput` contracts in `_base/model/` (exposed via `artifact_torch.nn`) for domain-specific data flow patterns, enabling type-safe callback development and static compatibility verification. Note that `ModelOutput` carries the optional `t_loss` entry consumed by the training loop.

**Callback Development**: Callback base classes live in `_base/components/callbacks/` (`model_io.py`, `forward_hook.py`, `backward_hook.py`, plus `checkpoint.py` for export callbacks); concrete implementations live in `_impl/callbacks/` (e.g. `model_io/loss.py` for `LossCallback`). Both are exposed via `artifact_torch.nn.callbacks` and its `model_io`/`forward_hook`/`backward_hook` submodules. New callbacks inherit from the appropriate base class and implement the required hook methods.

**Plan Development**: Plan bases (`ModelIOPlan`, `ForwardHookPlan`, `BackwardHookPlan` with their build contexts) live in `_base/components/plans/` and are exposed via `artifact_torch.nn.plans`. Plans group callbacks of a given kind: users implement classmethod hooks (`_get_score_callbacks(context)`, `_get_plot_callbacks(context)`, ...) that receive a build context exposing tracking writers. Plans are always declared as classes and instantiated by the framework via `Plan.build(context=...)`.

**Routine Development**: Routine bases live in `_base/components/routines/` and are exposed via `artifact_torch.nn.routines` (`DataLoaderRoutine`, `TrainDiagnosticsRoutine`) and `artifact_torch.spi` (`ArtifactRoutine`). `DataLoaderRoutine` builds its plans per `DataSplit` (hooks receive a `data_split` argument) and is constructed with `DataSplit`-keyed data loaders; `TrainDiagnosticsRoutine` attaches its plans to the model during training batches and its hooks are split-agnostic. The trainer groups injected routines in a `RoutineSuite` (`_base/trainer/routine_suite.py`) and executes them post-epoch.

**Model Tracker Development**: Extend `ModelTracker[ModelTrackingCriterionT]` in `_base/components/model_tracking/` (exposed via `artifact_torch.nn.model_tracking`) with domain-specific best-model tracking criteria (see `SingleScoreTracker`/`SingleScoreCriterion` for reference).

**Early Stopping Criteria**: The `EarlyStopper[StopperUpdateDataT]` base lives in `_base/components/early_stopping/`, with concrete implementations in `_impl/early_stopping/` (e.g. `EpochBoundStopper`, `ScoreMinimizationStopper`, `ScoreMaximizationStopper`); both are exposed via `artifact_torch.nn.early_stopping`. New stoppers subclass the base (or an intermediate base like `PatienceStopper`/`SingleScoreStopper`) and implement the termination logic.
