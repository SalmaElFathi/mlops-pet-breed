from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from prodml.config import settings


def _image_bytes(mode: str, fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, (64, 64)).save(buf, fmt)
    return buf.getvalue()


def _post(client: TestClient, data: bytes):
    return client.post("/predict", files={"file": ("upload", data, "application/octet-stream")})


def test_health_reports_a_loaded_model(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    assert r.json()["num_classes"] == 37


def test_predict_follows_the_response_contract(client: TestClient) -> None:
    r = _post(client, _image_bytes("RGB", "JPEG"))
    body = r.json()
    assert r.status_code == 200
    assert set(body) == {"breed", "species", "confidence", "top_3", "decision", "model_version"}
    assert len(body["top_3"]) == 3
    assert body["decision"] in {"confident", "uncertain"}
    assert body["top_3"][0]["breed"] == body["breed"]
    assert "x-request-id" in r.headers


@pytest.mark.parametrize(("mode", "fmt"), [("RGBA", "PNG"), ("L", "PNG"), ("CMYK", "TIFF")])
def test_unusual_image_modes_are_accepted(client: TestClient, mode: str, fmt: str) -> None:
    assert _post(client, _image_bytes(mode, fmt)).status_code == 200


def test_non_image_is_rejected_with_422(client: TestClient) -> None:
    assert _post(client, b"hello").status_code == 422


def test_oversized_upload_is_rejected_with_422(client: TestClient) -> None:
    assert _post(client, b"0" * (settings.max_upload_bytes + 1)).status_code == 422


def test_missing_file_gives_a_readable_422(client: TestClient) -> None:
    r = client.post("/predict")
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], list)


def test_unexpected_error_returns_500_without_traceback(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(image: Image.Image) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(client.app.state.predictor, "predict", boom)
    r = _post(client, _image_bytes("RGB", "JPEG"))
    assert r.status_code == 500
    assert r.json() == {"detail": "internal server error"}
    assert "boom" not in r.text
