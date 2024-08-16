## Applying CytoDL Models


```
pdm run cellsmap/model_features/apply_model.py apply_model --model_name mae --dataset_name 20240305_T01_001 --save_dir /your/path --overrides 
"{'key': 'value'}" 
```
`save_dir` defaults to the `results` directory. `overrides` can be used to change config parameters from the command line. 