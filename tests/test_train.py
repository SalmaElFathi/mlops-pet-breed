import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from prodml.model import PetClassifier
from prodml.train import (
    evaluate,
    extract_features,
    get_head,
    installed_packages,
    set_head,
    train_head,
)

CLASSES = ["a", "b", "c"]


def test_head_swap_for_both_families() -> None:
    for backbone in ("resnet18", "mobilenet_v3_small"):
        clf = PetClassifier(backbone, CLASSES, pretrained=False)
        original = get_head(clf.net, backbone)
        set_head(clf.net, backbone, nn.Identity())
        assert get_head(clf.net, backbone) is not original
        set_head(clf.net, backbone, original)
        assert get_head(clf.net, backbone) is original


def test_extract_features_keeps_the_head() -> None:
    clf = PetClassifier("resnet18", CLASSES, pretrained=False)
    original = get_head(clf.net, "resnet18")
    data = TensorDataset(torch.randn(4, 3, 224, 224), torch.tensor([0, 1, 2, 0]))
    feats, labels = extract_features(clf, DataLoader(data, batch_size=2))
    assert feats.shape == (4, 512)
    assert labels.tolist() == [0, 1, 2, 0]
    assert get_head(clf.net, "resnet18") is original


def test_train_head_learns_a_separable_problem() -> None:
    torch.manual_seed(0)
    labels = torch.randint(0, 3, (300,))
    feats = torch.nn.functional.one_hot(labels, 3).float() * 3 + torch.randn(300, 3) * 0.1
    head = train_head(feats, labels, 3, lr=0.05, batch_size=32, epochs=20, weight_decay=0.0, seed=0)
    accuracy = (head(feats).argmax(dim=1) == labels).float().mean().item()
    assert accuracy > 0.95


def test_evaluate_returns_every_track_c_metric() -> None:
    rng = np.random.default_rng(0)
    scores = evaluate(rng.normal(size=(50, 4)), rng.integers(0, 4, 50), threshold=0.7)
    expected = {
        "top1",
        "f1_macro",
        "ece_before",
        "ece",
        "temperature",
        "coverage",
        "selective_accuracy",
    }
    assert expected <= set(scores)


def test_installed_packages_lists_pins() -> None:
    assert "torch==" in installed_packages()
