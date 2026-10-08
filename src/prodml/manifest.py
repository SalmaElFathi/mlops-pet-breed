import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from prodml.data import load_data
from prodml.schemas import ManifestRecord
from prodml.config import settings
def build_records(split: str) -> list[ManifestRecord]:
    ds = load_data(settings.data_path, split, target_types=["category", "binary-category"])
    records: list[ManifestRecord] = []
    for i, path in enumerate(ds._images):
        img, (label, is_dog) = ds[i]  
        width, height = img.size
        records.append(
            ManifestRecord(
                image_id=Path(path).stem,
                path=Path(path).as_posix(),
                breed=ds.classes[label],
                species="dog" if is_dog else "cat",
                class_index=label, 
                split="test" if split == "test" else "train",
                width=width,
                height=height,
            )
        )
    return records


def main() -> None:
    trainval = build_records("trainval")
    test = build_records("test")

    breeds = sorted({r.breed for r in trainval + test})
    label_map = {breed: i for i, breed in enumerate(breeds)}
    settings.label_map_path.parent.mkdir(parents=True, exist_ok=True)
    settings.label_map_path.write_text(json.dumps(label_map, indent=2, ensure_ascii=False))

    idx = list(range(len(trainval)))
    y = [r.breed for r in trainval]
    _, val_idx = train_test_split(
        idx, test_size=settings.test_size, random_state=settings.random_state, stratify=y
    )
    for i in val_idx:
        trainval[i].split = "val"

    records = trainval + test
    for r in records:
        r.class_index = label_map[r.breed]

    settings.manifest_path.write_text(
        json.dumps([r.model_dump() for r in records], indent=2, ensure_ascii=False)
    )

    counts = {s: sum(r.split == s for r in records) for s in ("train", "val", "test")}
    print(counts)


if __name__ == "__main__":
    main()