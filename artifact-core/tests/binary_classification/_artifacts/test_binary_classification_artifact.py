import pytest
from artifact_core._base.core.hyperparams import NO_ARTIFACT_HYPERPARAMS, NoArtifactHyperparams
from artifact_core._base.typing.artifact_result import Score
from artifact_core._libs.resource_specs.binary_classification.protocol import (
    BinaryClassSpecProtocol,
)
from artifact_core._libs.resources.binary_classification.class_store import BinaryClassStore
from artifact_core._libs.resources.binary_classification.classification_results import (
    BinaryClassificationResults,
)
from artifact_core._libs.validation.classification.resource_validator import (
    ClassificationResourceValidator,
)
from artifact_core.binary_classification._artifacts.base import BinaryClassificationArtifact
from artifact_core.binary_classification._resources import BinaryClassificationArtifactResources
from pytest_mock import MockerFixture


class DummyBinaryClassificationArtifact(BinaryClassificationArtifact[NoArtifactHyperparams, Score]):
    def _evaluate_classification(
        self,
        true_class_store: BinaryClassStore,
        classification_results: BinaryClassificationResults,
    ) -> float:
        _ = true_class_store
        _ = classification_results
        return 1.0


@pytest.mark.unit
def test_resource_validation(
    mocker: MockerFixture,
    resource_spec: BinaryClassSpecProtocol,
    true_class_store: BinaryClassStore,
    classification_results: BinaryClassificationResults,
):
    patch_validate = mocker.patch.object(
        target=ClassificationResourceValidator,
        attribute="validate",
        wraps=ClassificationResourceValidator.validate,
    )
    artifact = DummyBinaryClassificationArtifact(
        resource_spec=resource_spec, hyperparams=NO_ARTIFACT_HYPERPARAMS
    )
    resources = BinaryClassificationArtifactResources.from_stores(
        true_class_store=true_class_store,
        classification_results=classification_results,
    )
    result = artifact.compute(resources=resources)
    patch_validate.assert_called_once_with(
        true_class_store=true_class_store,
        classification_results=classification_results,
    )
    assert result == 1.0
