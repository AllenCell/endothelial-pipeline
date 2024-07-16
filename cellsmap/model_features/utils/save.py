from bioio.writers import OmeTiffWriter
from pathlib import Path

class Save():
    def __init__(self, save_dir):
        self.save_dir= Path(save_dir)
        self.timepoint_counts = {}
    def __call__(self, image):
        tp = image.meta['T']
        if not self.timepoint_counts.get(tp):
            self.timepoint_counts[tp] = 0
        path = (self.save_dir/f'T{tp}_{self.timepoint_counts[tp]}.tif').resolve()
        OmeTiffWriter.save(image.detach().cpu().numpy().astype(float), path)
        self.timepoint_counts[tp] += 1
        image.meta.update({'crop_path': str(path)})
        return image
        