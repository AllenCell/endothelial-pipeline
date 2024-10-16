import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from PIL import Image
from cellsmap.util import io
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn import decomposition as skdecomp
import matplotlib.pyplot as plt

import ae

def get_vae_latent_variables(model, dataloader):
    model.eval()
    latent_vars = []
    with torch.no_grad():
        for batch_data in dataloader:
            import pdb; pdb.set_trace()
            mu, logvar = model.encoder(batch_data)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            latent_vars.append(z)
    return torch.cat(latent_vars, dim=0)


if __name__ == '__main__':

    model = ae.ConvAutoencoder(latent_dim=ae.N)
    checkpoint = torch.load('best_test_model.pth')
    model.load_state_dict(checkpoint)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    reader = io.load_dataset("20240305_T01_001", channels=["CDH5_Tubulin", "Nuc_Seg"], time_start=0, time_end=0, level=2)
    print(reader.shape)

    dataset = ZARRDatasetEval(reader=reader, crop_size=ae.M, transform=transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    latent_vars = get_latent_variables(model, data_loader)
    latent_vars = latent_vars.squeeze(-1).squeeze(-1)
    latent_vars_numpy = latent_vars.numpy()

    reader_late = io.load_dataset("20240305_T01_001", channels=["CDH5_Tubulin", "Nuc_Seg"], time_start=576, time_end=576, level=2)
    print(reader_late.shape)

    dataset_late = ZARRDatasetEval(reader=reader_late, crop_size=ae.M, transform=transform)
    data_loader_late = DataLoader(dataset_late, batch_size=64, shuffle=False)
    latent_vars_late = get_latent_variables(model, data_loader_late)
    latent_vars_late = latent_vars_late.squeeze(-1).squeeze(-1)
    latent_vars_numpy_late = latent_vars_late.numpy()

    latent_vars_numpy_combined = np.concatenate([latent_vars_numpy, latent_vars_numpy_late])

    pca = skdecomp.PCA(n_components=2)
    latent_vars_pca = pca.fit_transform(latent_vars_numpy_combined)

    def index_to_xy(idx, X=reader.shape[-1], Y=reader.shape[-2], s=ae.M):
        N_x = (X - s) // s + 1
        y = (idx // N_x) * s
        x = (idx % N_x) * s
        return y, x

    df = []
    # fov = reader[0].compute()
    for i, pcs in enumerate(latent_vars_pca):
        y, x = index_to_xy(i)
        #crop = fov[0, y:y+ae.M, x:x+ae.M]
        df.append({
        #    "x": x, "y": y, "pc1": pcs[0], "pc2": pcs[1], "mean_intensity": crop.mean(), "max_intensity": crop.max()
        "x": x, "y": y, "pc1": pcs[0], "pc2": pcs[1]
        })
    df = pd.DataFrame(df)

    fig, ax = plt.subplots(1,1)
    ax.scatter(df.x, df.y, c=df.pc1, s=10, cmap="jet")
    plt.savefig("pc1.png")

    fig, ax = plt.subplots(1,1)
    ax.scatter(df.x, df.y, c=df.pc2, s=10, cmap="jet")
    plt.savefig("pc2.png")

    fig, ax = plt.subplots(1,1)
    ax.scatter(df.pc1, df.pc2, c=df.mean_intensity, s=10, cmap="jet")
    plt.savefig("scatter.png")

    fig, ax = plt.subplots(1,1)
    ax.scatter(df.pc1, df.pc2, c=df.max_intensity, s=10, cmap="jet")
    plt.savefig("scatter_max.png")
