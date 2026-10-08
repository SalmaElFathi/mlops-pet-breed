# Data

## Source
Oxford-IIIT Pet (7,349 images, 37 breeds). Downloaded by torchvision into
`data/raw/oxford-iiit-pet/`. Images are not stored in Git (tracked by DVC).

## Splits
- `trainval` / `test` as shipped with the dataset.
- `val`: 20% of `trainval`, stratified by breed, seed 42.
- Result: train 2944, val 736, test 3669. The test set is not used before Session 4.

## Files
- `label_map.json`: 37 sorted breeds, committed. `class_index` values come from here.
- `manifest.json`: one record per image (schema: `src/prodml/schemas.py`).
- `manifest_corrupted.json`: same fields, with `corruption` and `severity` set.
- `corruptions.py`: generates the corrupted copies of the test set.

## Corruptions (applied to the test set, 3 severities)

| Corruption | Simulates | Sev. 1 | Sev. 2 | Sev. 3 |
|---|---|---|---|---|
| `gaussian_blur` | out-of-focus photo | radius 1 | radius 2 | radius 4 |
| `brightness_up` | direct sunlight | x1.3 | x1.6 | x2.0 |
| `brightness_down` | indoor evening | x0.7 | x0.5 | x0.3 |
| `jpeg` | messaging-app re-encode | quality 60 | quality 30 | quality 10 |
| `downscale` | old handset | 160x160 | 96x96 | 64x64 |
| `motion_blur` | camera motion (horizontal) | 5 px | 9 px | 15 px |

JPEG quality 30 and 96x96 are fixed by the course handbook (severity 2). The other
values are project choices. Corruptions are deterministic. They are used to test
drift detectors, not to train models.

## Regenerate

```bash
python -m prodml.manifest
python data/corruptions.py
```