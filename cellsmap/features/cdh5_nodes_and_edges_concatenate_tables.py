import pandas as pd
from pathlib import Path
from cellsmap.util.set_output import get_output_path

cdh5_nodes_and_edges_out_dir = Path(get_output_path('cdh5_nodes_and_edges'), verbose=False)
concatenated_table_out_dir = Path(get_output_path(Path(__file__).stem), verbose=False)

tables_alignments = Path(cdh5_nodes_and_edges_out_dir).glob(f'**/tables_alignments/*.csv')
tables_segprops = Path(cdh5_nodes_and_edges_out_dir).glob(f'**/tables_segmentation_properties/*.csv')

## lastly, concatenate the tables from each timepoint into a single output table
print('Concatenating individual timepoint tables together and saving...')
master_table = pd.concat([pd.read_csv(filepath) for filepath in tables_alignments])
master_table.to_csv(concatenated_table_out_dir / f'alignments.csv', index=False)

try:
    master_table = pd.concat([pd.read_csv(filepath) for filepath in tables_segprops])
    master_table.to_csv(concatenated_table_out_dir / f'segmentation_properties.csv', index=False)
except ValueError: # error that will be raised if there are no files found in tables_segprops
    print('No segmentation properties tables found. Skipping concatenation of segmentation properties tables.')

print('\N{fireworks} Done.')
print(f'Concatenated tables saved to {concatenated_table_out_dir}.')
