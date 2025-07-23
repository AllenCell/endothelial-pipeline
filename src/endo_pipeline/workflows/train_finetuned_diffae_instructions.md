# DiffAE Finetuning with Paired Live/Fixed data

## Creating the dataset
1. Running `uv run src/endo_pipeline/workflows/generate_csv_for_finetuning_diffae.py` will align, standard dev project, and create 2-channel composites of paired live/fixed or 20x/40x datasets that can be used for training. Outputs will be saved to `results/finetune_paired_dataset/{dataset_pair_type}`, where `dataset_pair_type: Literal["live_fixed","20X_40X"]`. If you have an intended train/test split, you can set `--split False` and then split the `dataset.csv` file into two new csvs in the same folder named `train.csv` and `val.csv`. Otherwise, the default is `split = True` for a random train/test split.
    - If `--dataset_pair_type live_fixed` is used, a hardcoded set of pre/post fixation data will be used to generate the `dataset.csv` file. This is the default setting.
    - If `--dataset_pair_type 20x_40x` is used, a hardcoded set of 20x/40x paired data will be used to generate the `dataset.csv` file.


## Finetuning the DiffAE
1. run `uv run src/endo_pipeline/workflows/train_finetuned_diffae_model.py` to finetune the `diffae_04_10` checkpoint for live/fixed pairs. The `dataset_pair_type` argument should be set to `20x_40x` to instead fintune the model for 20x/40x pairs. To finetune a different model checkpoint, pass in `--model_name {model_name}` in the command line, where `{model_name}` refers to one of the CytoDL models listed in `src/endo_pipeline/configs/models` directory.
