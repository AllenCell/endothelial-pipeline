from lightning.pytorch.callbacks import Callback
import pandas as pd
from pathlib import Path

class CSVSaver(Callback):
    def __init__(self, save_dir, meta_keys=[]):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.meta_keys = meta_keys

    def on_predict_epoch_end(self, trainer, pl_module):
        # Access the list of predictions from all predict_steps
        predictions = trainer.predict_loop.predictions
        feats = []
        for pred, meta in predictions:
            # exclude cls token
            batch_feats = pd.DataFrame(pred[1:].mean(axis=0))
            batch_feats['location'] = range(batch_feats.shape[0])
            batch_feats['time'] = meta['T']
            feats.append(batch_feats)
        pd.concat(feats).to_csv(self.save_dir / 'predictions.csv', index=False)