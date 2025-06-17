# DiffAE Finetuning with Paired Live/Fixed data

## Creating the dataset
1. To maintain compatibility with the zarrs for inference, we need to create a `MultiDimImageDataset` that references a csv containing paths to images with two channels - one for the aligned live data and one for the aligned fixed data. Running `uv run cellsmap/model_features/paired_data_validation.py --align_only True` will align the live and fixed data, creating a csv file named with the pattern `aligned_<fixed_dataset_name>_vs_aligned_<live_dataset_name>.csv` under the `results/<model_name>/` directory where `<model_name>` is by default "diffae_finetuned_for_fixed" for the pre/post fixation data and "diffae_04_10" for the 20x/40x paired data. Both of these model names should be updated when new models are trained. This .csv file will have two columns - `fixed` and `moving`. To match registration standard terminology, the `moving` images refer to the chemically fixed images (which will be registered), while the `fixed` column refers to the live images (the reference images to register onto).
2. Running `uv run cellsmap/model_features/paired_finetune/generate_paired_dataset.py --model_name <model_name> --dataset_type <dataset_type>` will combine and standard dev project these images into the two channel images that can be used for training. Outputs will be saved to `results/finetune_paired_dataset/<dataset_type>`. If you have an intended train/test split, you can split the `dataset.csv` file into two new csvs in the same folder named `train.csv` and `val.csv`. Otherwise, you can set `--split True` for a random train/test split.
    - If `--model_name diffae_finetuned_for_fixed --dataset_type live_fixed` is used, pre/post fixation data will be used to generate the `dataset.csv` file
    - If `--model_name diffae_04_10 --dataset_type 20x_40x` is used, 20x/40x paired data will be used to generate the `dataset.csv` file


## Finetuning the DiffAE
1. run `uv run cellsmap/model_features/paired_finetune/train_finetuned_model.py --model_name diffae_04_10 --dataset_type live_fixed` to finetune the `diffae_04_10` checkpoint. The `dataset_type` argument should be set to `20x_40x` to use the aligned 20x/40x data for training.
3. Once this model has been trained, add its mlflow run id to `model_config.yaml`.
