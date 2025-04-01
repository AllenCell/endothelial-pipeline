from pathlib import Path
from bioio import BioImage
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from cellsmap.util import cdh5_preprocessing as preproc, dataset_io

def get_nuclei_count_from_image_filepath(filepath: Path, T=None, C=None) -> int:
    img = BioImage(filepath)
    img_arr = img.get_image_data('TCZYX', T=T, C=C).squeeze()
    num_nuclei = (np.unique(img_arr) > 0).size
    return num_nuclei


prj_dir = dataset_io.get_prj_dir(is_test=False)
out_dir = dataset_io.get_results_dir(Path(__file__).stem, is_test=False)
img_dir = prj_dir / 'results' / 'generate_label_free_nuc_pred' / '20241120_20X'
annotation_dir = prj_dir / 'results' / 'generate_label_free_nuc_pred' / '20241120_20X_missed_nuclei'
Path.mkdir(out_dir, exist_ok=True, parents=True)

# Load the table of missed nuclei counts (manually annotated)
missed_nuc_df = pd.read_csv(annotation_dir / 'missed_nuclei_count_table.csv')

# Load the label free nuclei predictions
pred_paths = {fp: fp.stem for fp in img_dir.glob('*.tif*')}
nuclei_predictions = {fp: BioImage(fp) for fp in pred_paths}

# Build a table containing the number of predicted nuclei per timepoint
# first get number of nuclei for each image in pred_paths
nuclei_counts = {}
for fp in pred_paths:
    # NOTE: I saved the nuclei predictions to the third Channel (C=2) and
    # each image was saved with a single timepoint (T=0)
    nuclei_counts[fp] = get_nuclei_count_from_image_filepath(fp, T=0, C=2)

# build a new dataframe with the number of nuclei, the filename, and the timepoint
cols = ['T', 'num_nuclei_predicted', 'image_filepath']
data = [(preproc.extract_T(fp), c, fp) for fp, c in nuclei_counts.items()]
nuclei_counts_df = pd.DataFrame(data, columns=cols)

# the values in the 'T' column of missed_nuc_df are incorrect, so reassign
# them column based on the 'Name' column
missed_nuc_df['T'] = missed_nuc_df['Name'].apply(lambda x: preproc.extract_T(x))

# add the number of missed nuclei at each timepoint to nuclei_counts_df
num_missed_nuc_per_timepoint_map = dict(zip(missed_nuc_df['T'], missed_nuc_df['Points']))
nuclei_counts_df['dataset_name'] = nuclei_counts_df['image_filepath'].transform(lambda x: x.parent.stem)
nuclei_counts_df['missed_nuclei'] = nuclei_counts_df['T'].transform(lambda t: num_missed_nuc_per_timepoint_map[t])
nuclei_counts_df['total_nuclei'] = nuclei_counts_df['num_nuclei_predicted'] + nuclei_counts_df['missed_nuclei']
nuclei_counts_df['percent_nuclei_detected'] = nuclei_counts_df.apply(lambda row: row['num_nuclei_predicted'] / row['total_nuclei'], axis=1)

# rearrange the order of the columns for readability
nuclei_counts_df = nuclei_counts_df[['dataset_name',
                                     'T',
                                     'num_nuclei_predicted',
                                     'missed_nuclei',
                                     'total_nuclei',
                                     'percent_nuclei_detected',
                                     'image_filepath']]

# finally, plot the change in the number of nuclei over time
dataset_name = nuclei_counts_df['dataset_name'].iloc[0]
fig, ax = plt.subplots()
sns.lineplot(data=nuclei_counts_df, x='T', y='num_nuclei_predicted', marker='.', ax=ax, label='Predicted Nuclei')
sns.lineplot(data=nuclei_counts_df, x='T', y='missed_nuclei', marker='.', ax=ax, label='Missed Nuclei')
sns.lineplot(data=nuclei_counts_df, x='T', y='total_nuclei', marker='.', ax=ax, label='Total Nuclei')
ax.set_xlabel('Timeframe')
ax.set_ylabel('Number of Nuclei')
ax.set_title('Number of Nuclei Over Time')
plt.tight_layout()
plt.savefig(out_dir / f'{dataset_name}_nuclei_counts_over_time.png', bbox_inches='tight', dpi=300)
plt.close(fig)

fig, ax = plt.subplots()
sns.lineplot(data=nuclei_counts_df, x='T', y='percent_nuclei_detected', marker='.', c='k', ax=ax, label='Total Nuclei')
ax.set_xlabel('Timeframe')
ax.set_ylabel('Number of Nuclei')
ax.set_title('Percent of Nuclei Detected Over Time')
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(out_dir / f'{dataset_name}_percent_nuclei_detected_over_time.png', bbox_inches='tight', dpi=300)
plt.close(fig)
