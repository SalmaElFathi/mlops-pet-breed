from unittest.mock import MagicMock, patch

import pytest
from mlflow.exceptions import MlflowException

from prodml import registry


def _fake_client(prod_top1: float | None, cand_top1: float) -> MagicMock:
    client = MagicMock()
    scores = {"cand": cand_top1, "prod_run": prod_top1}
    client.get_run.side_effect = lambda rid: MagicMock(
        data=MagicMock(metrics={"top1": scores[rid]})
    )
    prod = MagicMock(version="3", run_id="prod_run")

    def by_alias(name: str, alias: str):
        if alias == "production" and prod_top1 is not None:
            return prod
        raise MlflowException("alias not found")

    client.get_model_version_by_alias.side_effect = by_alias
    client.search_model_versions.return_value = []
    return client


def test_better_candidate_goes_to_staging():
    client = _fake_client(prod_top1=0.93, cand_top1=0.95)
    with (
        patch.object(registry, "_client", return_value=client),
        patch("mlflow.register_model", return_value=MagicMock(version="4")),
    ):
        assert registry.promote_if_better("cand") is True
    client.set_registered_model_alias.assert_called_once_with(registry.MODEL_NAME, "staging", "4")


def test_worse_candidate_is_rejected_without_registering():
    client = _fake_client(prod_top1=0.95, cand_top1=0.90)
    with (
        patch.object(registry, "_client", return_value=client),
        patch("mlflow.register_model") as register,
    ):
        assert registry.promote_if_better("cand") is False
    register.assert_not_called()
    client.set_registered_model_alias.assert_not_called()


def test_margin_is_respected():
    client = _fake_client(prod_top1=0.95, cand_top1=0.955)
    with patch.object(registry, "_client", return_value=client):
        assert registry.promote_if_better("cand", margin=0.01) is False


def test_first_model_is_accepted_when_no_production():
    client = _fake_client(prod_top1=None, cand_top1=0.90)
    with (
        patch.object(registry, "_client", return_value=client),
        patch("mlflow.register_model", return_value=MagicMock(version="1")),
    ):
        assert registry.promote_if_better("cand") is True


def test_auto_promotion_to_production_is_forbidden():
    with pytest.raises(ValueError):
        registry.promote_if_better("cand", alias="production")
