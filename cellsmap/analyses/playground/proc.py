import numpy as np
import cellsmap.analyses.playground.const as const

def normalize_zplane(zplane, global_norm=True):
    vmin, vmax = const.VMIN, const.VMAX
    if not global_norm:
        vmin, vmax = np.percentile(zplane.flatten(), [5, 95])
    zplane = np.clip(zplane, vmin, vmax)
    zplane = (zplane-vmin) / (vmax-vmin)
    return zplane

def normalize_crop(crop, global_norm=True):
    single_channel = False
    if crop.ndim == 2:
        single_channel = True
        crop = crop.reshape(-1, *crop.shape)
    crop_norm = np.zeros_like(crop)
    for z, zplane in enumerate(crop):
        crop_norm[z] = normalize_zplane(zplane, global_norm=global_norm)
    crop = crop_norm
    if single_channel:
        crop = crop.squeeze()
    return crop