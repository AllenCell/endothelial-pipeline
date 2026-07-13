# Dynamics of ML-based morphological features indicate a shear stress-dependent bifurcation of hiPSC-derived endothelial cell states

The code in this repository runs all analyses and generates all figures, tables, and movies for [Angelini, Leveille, Parent, and Zaunbrecher et al](https://doi.org/10.64898/2026.07.07.736803).
It is primarily intended to support reproducibility of our research.
In addition, researchers may find parts of this code valuable for future work.

> [!NOTE]
> The `main` branch reflects the most up-to-date version of the code.
> To exactly reproduce the contents from the [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.07.07.736803v1) version of the manuscript, use the [`bioRxiv-v1` release](https://github.com/AllenCell/endothelial-pipeline/releases/tag/bioRxiv-v1).

We release all timelapse data used in this study in the OME-Zarr format to democratize their access.
The data are publicly available on [S3](https://open.quiltdata.com/b/allencell/tree/aics/endo_cell_state_dynamics/) under the [Allen Institute for Cell Science Terms of Use](https://www.allencell.org/terms-of-use.html).
The data are also available via the AWS S3 API directly at `s3://allencell/aics/endo_cell_state_dynamics`.

To facilitate data exploration, all datasets are accessible through [BioFile Finder (BFF)](https://bff.allencell.org/).
BFF is an open-use web application that enables rich metadata search, filtering, and sorting of datasets that can be downloaded or viewed directly.

#### ▌ Access all [Endothelial cell state dynamics datasets](https://bff.allencell.org/app?c=File+Name%3A45%2CAnalysis+Workflow%3A30%2CBiological+Entity%3A34%2CCell+Line%3A45%2CChannel+0+Image+Content%3A35%2CChannel+0+Laser+Power+%28mW%29%3A24&group=Data+Type&group=Dataset&group=Shear+Stress+Bin+%28dyn%2Fcm%C2%B2%29&group=Date&source=%7B%22name%22%3A%22all_datasets_manifest.csv+%287%2F6%2F2026+10%3A27%3A29+PM%29%22%2C%22type%22%3A%22csv%22%2C%22uri%22%3A%22s3%3A%2F%2Fallencell%2Faics%2Fendo_cell_state_dynamics%2Fall_datasets_manifest.csv%22%7D) on BFF

Individual timelapses can be viewed using [Volume Explorer (Vol-E)](https://volumeviewer.allencell.org) through BFF by right clicking on any image data entry and selecting _Open with_ > _Vol-E_.
Users can additionally interactively explore the image and feature data for grid-based patches and cell segmentations simultaneously using [Timelapse Feature Explorer (TFE)](https://tfe.allencell.org/).

#### ▌ Explore [VE-cadherin segmentations for endothelial cell state dynamics](https://timelapse.allencell.org/viewer?collection=https%3A%2F%2Fallencell.s3.amazonaws.com%2Faics%2Fendo_cell_state_dynamics%2Ftimelapse_feature_explorer_cdh5%2Fcollection.json&dataset=6+dyn%2Fcm%C2%B2+%2820250618%2C+P0%29&feature=orientation&bg-key=backdrop_gfp_max_proj&t=110&color=colorcet-cet_c8&keep-range=0&range=0%2C3.142&palette-key=adobe&filters=not_steady_state%3A%3Affd%2Ccell_piling%3A%3Affd&seg=1&path=1&path-color=ff00ff&path-width=1.500&path-ramp=esri-blue_red_8&path-mode=0&path-breaks=0&path-steps=25%21%2C0&path-persist=0&path-overlay=30&scalebar=1&timestamp=1&filter-color=dddddd&filter-mode=1&outlier-color=c0c0c0&outlier-mode=1&outline-color=ff00ff&outline-mode=0&outline-palette-key=neon_reordered&edge=1&edge-color=00000080&centroids=0&centroid-mode=0&centroid-color=aaaaaa&centroid-radius=4&tab=scatter_plot&interpolate=1&scatter-x=time_hours&scatter-y=orientation&sc-hist=1&scatter-bins=20&scatter-range=all&sc-cont=0&sc-cont-num=10&sc-avg=0&sc-avg-n=5&sc-avg-w=1.6&bg=1&bg-brightness=100&bg-sat=100&fg-alpha=50&vc=0&vc-key=_motion_&vc-color=000000&vc-scale=5&vc-thickness-scaling=0&vc-thickness=1&vc-tooltip=c&vc-time-int=5&p3d-x=polar_theta&p3d-y=polar_r&p3d-z=rho&p3d-vc-bins=20&p3d-vc-subs=1&p3d-vc-scale=1&p3d-vc-ramp=matplotlib-viridis%21&p3d-vc-thresh=5&p3d-avg-w=3&p3d-avg-n=1&p3d-gauss=0&p3d-gauss-bw=15) on TFE

#### ▌ Explore [Grid-based patches for endothelial cell state dynamics](https://timelapse.allencell.org/viewer?collection=https%3A%2F%2Fallencell.s3.amazonaws.com%2Faics%2Fendo_cell_state_dynamics%2Ftimelapse_feature_explorer_grid%2Fcollection.json&dataset=6+dyn%2Fcm%C2%B2+%2820250618%2C+P0%29&feature=polar_theta&bg-key=backdrop_gfp_max_proj&t=110&color=colorcet-cet_c2&keep-range=0&range=0%2C3.142&palette-key=adobe&filters=not_steady_state%3A%3Affd%2Ccell_piling%3A%3Affd&seg=1&path=0&path-color=ff00ff&path-width=1.500&path-ramp=esri-blue_red_8&path-mode=0&path-breaks=0&path-steps=25%21%2C0&path-persist=0&path-overlay=30&scalebar=1&timestamp=1&filter-color=dddddd&filter-mode=1&outlier-color=c0c0c0&outlier-mode=1&outline-color=ff00ff&outline-mode=0&outline-palette-key=neon_reordered&edge=1&edge-color=00000040&centroids=0&centroid-mode=0&centroid-color=aaaaaa&ct-alpha=50&centroid-radius=4&tab=scatter_plot&interpolate=1&scatter-x=time_hours&scatter-y=polar_theta&sc-hist=0&scatter-bins=20&scatter-range=all&sc-cont=0&sc-cont-num=10&sc-avg=0&sc-avg-n=5&sc-avg-w=1.6&bg=1&bg-brightness=100&bg-sat=100&fg-alpha=50&vc=0&vc-key=_motion_&vc-color=000000&vc-scale=5&vc-thickness-scaling=0&vc-thickness=1&vc-tooltip=c&vc-time-int=5&p3d-x=polar_theta&p3d-y=polar_r&p3d-z=rho&p3d-vc-bins=10&p3d-vc-subs=1&p3d-vc-scale=1&p3d-vc-ramp=matplotlib-viridis%21&p3d-vc-thresh=5&p3d-avg-w=3&p3d-avg-n=5&p3d-gauss=0&p3d-gauss-bw=15) on TFE

## Installation

This project requires Python 3.11.
We recommend using the most recent version of Python 3.11 (Python 3.11.12).
Package dependencies can be found in the `pyproject.toml` file.

### Installation using UV

We recommend using the Python package manager [uv](https://docs.astral.sh/uv) to manage dependencies and virtual environments.
Install uv following their [installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

**1. Navigate to where you want to clone this repository**

```bash
cd /path/to/directory/
```

**2. Clone the repo from GitHub**

```bash
git clone git@github.com:AllenCell/endothelial-pipeline.git
cd endothelial-pipeline
```

**3. Install the dependencies using uv**

For basic installation with just the core dependencies:

```bash
uv sync --no-dev
```

If you plan to develop code, you should also install the development dependencies:

```bash
uv sync
```

If you are on the Allen Institute for Cell Science local network, you can load on-prem data by installing `aicsfiles` (which is included in the optional `internal` dependency group):

```bash
uv sync --extra internal
```

**4. Activate the virtual environment**

Activate the virtual environment in the terminal:

For Windows:

```powershell
\path\to\venv\Scripts\activate
```

For Linux/Mac:

```bash
source /path/to/venv/bin/activate
```

You can deactivate the virtual environment using:

```
deactivate
```

### Alternative installation using `pip`

This project also includes a `requirements.txt` generated from the `uv.lock` file, which can be used to install requirements using `pip`.

> [!NOTE]
> This installation method will only install core dependencies.
> We recommend using uv to handle more complex installations of development and optional dependencies.

**1. Navigate to where you want to clone this repository**

```bash
cd /path/to/directory/
```

**2. Clone the repo from GitHub**

```bash
git clone git@github.com:AllenCell/endothelial-pipeline.git
cd endothelial-pipeline
```

**3. Create and activate a new virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
```

**4. Install the dependencies using pip**

```bash
pip install -r requirements.txt
```

**5. Install the package**

```bash
pip install -e .
```

## Usage

All workflows in this repository can be run using the `endopipe` command.
Running the command without any arguments will provide a list of all available workflows.

```bash
uv run endopipe
```

You may want to use tags to filter these workflows to specific areas.

```bash
# Show all available tags
uv run endopipe -t

# Show only workflows with given tag
uv run endopipe -f TAG
```

Run workflows by passing the name of the workflow to the `endopipe` command.
All workflows have additional details on usage and arguments, which you can access by appending the `-h` or `--help` flag.

> [!IMPORTANT]
> Most workflows provided are designed to run in a high-performance computing setting or need the use of GPUs to run in a reasonable amount of time.
> To support review of the code, all production workflows have a "demo mode" (which you can access by appending the `-d` or `--demo-mode` flag) that modifies (e.g. by reducing the number of iterations) the workflow to run in a shorter amount of time.

```bash
# Run the workflow
uv run endopipe NAME-OF-WORKFLOW

# Show the workflow help message
uv run endopipe NAME-OF-WORKFLOW -h

# Run the workflow in demo mode
uv run endopipe NAME-OF-WORKFLOW -d
```

### Reproduce figures, tables, and movies

To reproduce figures, tables, and movies from the manuscript, run the corresponding workflow.
Some workflows should be run with an NVIDIA GPU, as indicated by the workflow help message.
Run the workflow with the GPU flag (`-g` or `--num-gpus`) to make sure GPUs are visible to the workflow.
All workflows will run without a GPU, but will be noticeably slower.

```bash
# Reproduce main figure N
uv run endopipe figure-N

# Reproduce supplemental figure N
uv run endopipe supp-figure-N

# Reproduce all supplemental movies
uv run endopipe supp-movies

# Reproduce all supplemental tables
uv run endopipe supp-tables
```
