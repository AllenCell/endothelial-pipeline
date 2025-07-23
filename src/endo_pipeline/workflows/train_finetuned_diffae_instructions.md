# DiffAE Finetuning with Paired Live/Fixed data

## Creating the dataset
1. Running `uv run src/endo_pipeline/workflows/generate_csv_for_finetuning_diffae.py --dataset_type <dataset_type>` will align, standard dev project, and create 2-channel composites of paired live/fixed or 20x/40x datasets that can be used for training. Outputs will be saved to `results/finetune_paired_dataset/<dataset_type>`. If you have an intended train/test split, you can set `--split False` and then split the `dataset.csv` file into two new csvs in the same folder named `train.csv` and `val.csv`. Otherwise, the default is `split=True` for a random train/test split.
    - If `--dataset_pair_type live_fixed` is used, a hardcoded set of pre/post fixation data will be used to generate the `dataset.csv` file. To override this, pass `--fixed_datasets` and `--moving_datasets` with the paths to paired dataset names
    - If `--dataset_pair_type 20x_40x` is used, a hardcoded set of 20x/40x paired data will be used to generate the `dataset.csv` file. To override this, pass `--fixed_datasets` and `--moving_datasets` with the paths to paired dataset names


## Finetuning the DiffAE
1. run `uv run src/endo_pipeline/workflows/train_finetuned_diffae_model.py --model_name diffae_04_10 --dataset_pair_type live_fixed` to finetune the `diffae_04_10` checkpoint. The `dataset_type` argument should be set to `20x_40x` to use the aligned 20x/40x data for training.
