from pathlib import Path
from bioio import BioImage
import matplotlib.pyplot as plt
import pandas as pd
from cellsmap.util import cdh5_preprocessing as preproc, io

prj_dir = Path(__file__).parents[2]
out_dir = prj_dir / 'results' / Path(__file__).stem
img_dir = prj_dir / 'results' / 'generate_label_free_nuc_pred' / '20241120_20X'
annotation_dir = prj_dir / 'results' / 'generate_label_free_nuc_pred' / '20241120_20X_missed_nuclei'

# Load the table of missed nuclei counts (manually annotated)
missed_nuc_df = pd.read_csv(annotation_dir / 'missed_nuclei_count_table.csv')

# Load the label free nuclei predictions
pred_paths = {fp: fp.stem for fp in img_dir.glob('*.tif*')}
nuclei_predictions = {fp: BioImage(fp) for fp in pred_paths}

# Build a table containing the number of predicted nuclei per timepoint

