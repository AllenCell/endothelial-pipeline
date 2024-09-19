# Workflows for fitting stochastic dynamical model to time series data of endothelial cells

- set up model configurations in dynamics_config.yaml (number of bins for discretizing state space, analyze PCs versus original features, number of dimensions (1 or 2), save directory, etc.)
- if working on server with GPUs, select device to use via `export CUDA_VISIBLE_DEVICES=[device ID]` where `[device ID]` is the approprate ID for the desired GPU device
- in `cellsmap` git repo, run `pdm run cellsmap/analyses/workflows/fit_SDE_model.py`
    - user inputs via command line: config to use from `dynamics_config.yaml`, dataset to analyze (from `data_config.yaml`), and feature set to analyze (ML models trained on original image data, single crop trajectories in latent feature space)
    - use the plots in `[savedir]/figs/select_lag_[flow]` -- where `[savedir]` is the save directory pulled from `dynamics_config.yaml` and `[flow]` is the flow condition (relevant if fitting flow conditions separately) -- to select the appropriate time step lag to feed into the Langevin Regression algorithm (done for each flow condition if fitting separately)

- look at cost function plot to determine optimal sparsity level (`[savedir]/figs/cost_function_plot_[flow]`)

- once `fit_SDE_model.py` has finished running, in the `cellsmap` git repo run `pdm run cellsmap/analyses/workflows/analyze_SDE_model.py`
    - user inputs via command line: config used to fit model (where to pull saved model outputs from), number of terms to retain in the SINDy regression expressions for the drift and diffusion (see above step)
    - outputs phase portrait plot (phase line if 1D, phase plane if 2D), comparison of histogram and model's stationary density, plot of generalized stationary potential with and without the gradient/flux vector decomposition
