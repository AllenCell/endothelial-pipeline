from lightning.pytorch.callbacks import Callback
import pandas as pd
from pathlib import Path

class CSVSaver(Callback):
    def __init__(self, save_dir, meta_keys=[]):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.meta_keys = meta_keys

    def to_dataframe(self, x):
        return pd.DataFrame(x)

    def on_predict_epoch_end(self, trainer, pl_module):
        # Access the list of predictions from all predict_steps
        predictions = trainer.predict_loop.predictions
        feats = []
        for pred, meta in predictions:
            # exclude cls token
            batch_feats = self.to_dataframe(pred)
            batch_feats['crop_index'] = range(len(batch_feats))
            for k in self.meta_keys:
                batch_feats[k] = meta[k]
            feats.append(batch_feats)
        pd.concat(feats).to_csv(self.save_dir / 'predictions.csv', index=False)

class CLSSaver(CSVSaver):
    """Save CLS token only
    """
    def to_dataframe(self, x):
        return pd.DataFrame(x[1])

class MAESaver(CSVSaver):
    def to_dataframe(self, x):
        return pd.DataFrame(x[1:].mean(axis=0))
    
class JEPASaver(CSVSaver):
    def to_dataframe(self, x):
        return pd.DataFrame(x.mean(axis=1))
    
