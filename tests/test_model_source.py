from unittest.mock import MagicMock, patch

import pytest

from prodml import model_source


def test_local_fallback_when_registry_is_off():
    with (
        patch.object(model_source.settings, "use_registry", False),
        patch.object(model_source.PetClassifier, "load", return_value="model"),
    ):
        model, version = model_source.load_model()
    assert model == "model"
    assert version == model_source.settings.model_version


def test_registry_mode_loads_the_version_behind_the_alias(tmp_path):
    version = MagicMock(version="3", source="runs:/abc/model.pt", run_id="abc")
    client = MagicMock()
    client.get_model_version_by_alias.return_value = version
    with (
        patch.object(model_source.settings, "use_registry", True),
        patch.object(model_source.settings, "model_cache_dir", tmp_path),
        patch.object(model_source, "MlflowClient", return_value=client),
        patch.object(model_source.mlflow, "set_tracking_uri"),
        patch.object(
            model_source.mlflow.artifacts,
            "download_artifacts",
            return_value=str(tmp_path / "model.pt"),
        ),
        patch.object(model_source.PetClassifier, "load", return_value="model"),
    ):
        model, served_version = model_source.load_model()
    assert model == "model"
    assert served_version == "3"
    client.get_model_version_by_alias.assert_called_once_with("PetBreedClassifier", "production")


def test_registry_unreachable_raises_instead_of_serving_something_else():
    client = MagicMock()
    client.get_model_version_by_alias.side_effect = ConnectionError("down")
    with (
        patch.object(model_source, "MlflowClient", return_value=client),
        patch.object(model_source.mlflow, "set_tracking_uri"),
    ):
        with pytest.raises(RuntimeError):
            model_source.load_from_registry(retries=2, delay_s=0)
