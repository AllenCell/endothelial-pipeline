# Custom CLI types

The `endo_pipeline.cli` module provides custom [cyclopts](https://cyclopts.readthedocs.io/) type annotations for workflow parameters.

---

- [List types](#list-types)
- [`Datasets`](#datasets)
- [`CropPattern`](#croppattern)

---

## List types

`StrList`, `IntList`, and `FloatList` allow a parameter to accept multiple space- or flag-separated values from the CLI. The `UniqueStrList` and `UniqueIntList` variants additionally deduplicate and sort the values.

```python
from endo_pipeline.cli import UniqueStrList

def main(names: UniqueStrList) -> None:
    print(names)
```

```bash
# equivalent ways to pass a list
endopipe workflow a b c
endopipe workflow --names a --names b --names c
endopipe workflow --names a b c

# duplicates are removed and the list is sorted
endopipe workflow b c c a  # → ['a', 'b', 'c']
```

## `Datasets`

Accepts one or more dataset names **or** dataset collection names. Collections are automatically expanded into individual dataset names, duplicates are removed, and all names are validated against available dataset configs.

```python
from endo_pipeline.cli import Datasets
from endo_pipeline.configs import load_dataset_config

def main(datasets: Datasets) -> None:
    dataset_configs = [load_dataset_config(d) for d in datasets]
```

```bash
endopipe workflow --datasets dataset1 dataset2
endopipe workflow --datasets collection_name            # expanded automatically
endopipe workflow --datasets dataset1 collection_name   # mix of both
```

## `CropPattern`

Accepts a crop pattern string (`"grid"` or `"tracked"`). Input is converted to lowercase before validation.

```python
from endo_pipeline.cli import CropPattern

def main(crop_pattern: CropPattern) -> None:
    print(crop_pattern)
```

```bash
endopipe workflow --crop-pattern grid
endopipe workflow --crop-pattern Grid   # converted to lowercase automatically
endopipe workflow --crop-pattern other  # raises an error
```
