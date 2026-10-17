from prodml.sweep import sample_configs

BASE = {"backbone": "resnet18", "seed": 42}


def test_sample_configs_is_reproducible() -> None:
    assert sample_configs(BASE, 10, 0) == sample_configs(BASE, 10, 0)


def test_sample_configs_stay_in_the_search_space() -> None:
    configs = sample_configs(BASE, 50, 1)
    assert len(configs) == 50
    for c in configs:
        assert 1e-4 <= c["lr"] <= 1e-2
        assert c["batch_size"] in (16, 32, 64, 128)
        assert c["epochs"] in (5, 10, 20)
        assert c["backbone"] == "resnet18"
