import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from cellsmap.analyses.playground import const, proc

class ZARRDataset(Dataset):
    def __init__(self, reader, transform=None, debug=False):
        self.debug = debug
        self.reader = reader
        self.crop_size = const.M
        self.transform = transform
        self._generate_crops()

    def _generate_crops(self):
        data = []
        nsamples_per_timepoint = 50
        s = self.crop_size
        T, C, Y, X = self.reader.shape
        for t in range(1 if self.debug else T):
            fov = self.reader[t].compute()
            # TODO: Make flexible for nuclear seg in any channel
            msk = fov[-1].astype(int)
            # Prevent crops from being too close to the edge
            msk[:, :s], msk[:, -s:], msk[:s, :], msk[-s:, :] = 0, 0, 0, 0
            msk = msk.astype(int)
            labels_available = np.unique(msk[msk>0].flatten())
            labels = np.random.choice(labels_available, size=nsamples_per_timepoint)
            for label in labels:
                yc, xc = [int(u.mean()) for u in np.where(msk==label)]
                yi = yc - int(s/2)
                yf = yi + s
                xi = xc - int(s/2)
                xf = xi + s
                crop = fov[0][yi:yf, xi:xf]
                crop = proc.normalize_crop(crop, global_norm=False)
                data.append(crop)
        self.data = np.array(data)
        print("Dataset ready!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = Image.fromarray(np.uint8(image * 255))  # Convert numpy array to PIL image
        if self.transform:
            image = self.transform(image)
        return image

# TODO: Merge ZARRDataset and ZARRDatasetEval
class ZARRDatasetEval(Dataset):
    def __init__(self, reader, transform=None):
        self.reader = reader
        self.crop_size = const.M
        self.transform = transform
        self._generate_crops()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = Image.fromarray(np.uint8(image * 255))
        if self.transform:
            image = self.transform(image)
        return image

    def _generate_crops(self):
        data = []
        self._xcoord, self._ycoord = [], []
        self._mean_int, self._max_int = [], []
        s = self.crop_size
        T, C, Y, X = self.reader.shape
        for t in range(T):
            fov = self.reader[t].compute()
            # TODO: Make flexible for nuclear seg in any channel
            msk = fov[-1].astype(int)
            # Prevent crops from being too close to the edge
            msk[:, :s], msk[:, -s:], msk[:s, :], msk[-s:, :] = 0, 0, 0, 0
            msk = msk.astype(int)
            labels = np.unique(msk[msk>0].flatten())
            for label in labels:
                yc, xc = [int(u.mean()) for u in np.where(msk==label)]
                yi = yc - int(s/2)
                yf = yi + s
                xi = xc - int(s/2)
                xf = xi + s
                crop = fov[0][yi:yf, xi:xf]
                self._max_int.append(crop.max())
                self._mean_int.append(crop.mean())
                crop = proc.normalize_crop(crop, global_norm=False)
                self._xcoord.append(xi)
                self._ycoord.append(yi)
                data.append(crop)
        self.data = np.array(data)
        print(f"Dataset ready! Shape: {self.data.shape}")

    def get_crop_coordinates(self):
        return self._xcoord, self._ycoord

    def get_crop_intensity_properties(self):
        return self._mean_int, self._max_int

class PairZARRDataset(Dataset):
    def __init__(self, reader, dt=1, transform=None, debug=False):
        self._dt = dt
        self.debug = debug
        self.reader = reader
        self.crop_size = const.M
        self.transform = transform
        self._generate_crops()

    def _generate_crops(self):
        data = []
        nsamples_per_timepoint = 50
        s = self.crop_size
        T, C, Y, X = self.reader.shape
        for t in range(1 if self.debug else T-self._dt):
            fov_curr = self.reader[t].compute()
            fov_next = self.reader[t+self._dt].compute()
            # TODO: Make flexible for nuclear seg in any channel
            msk = fov_curr[-1].astype(int)
            # Prevent crops from being too close to the edge
            msk[:, :s], msk[:, -s:], msk[:s, :], msk[-s:, :] = 0, 0, 0, 0
            msk = msk.astype(int)
            labels_available = np.unique(msk[msk>0].flatten())
            labels = np.random.choice(labels_available, size=nsamples_per_timepoint)
            for label in labels:
                yc, xc = [int(u.mean()) for u in np.where(msk==label)]
                yi = yc - int(s/2)
                yf = yi + s
                xi = xc - int(s/2)
                xf = xi + s
                crop_curr = fov_curr[0][yi:yf, xi:xf]
                crop_curr = proc.normalize_crop(crop_curr, global_norm=True)
                crop_next = fov_next[0][yi:yf, xi:xf]
                crop_next = proc.normalize_crop(crop_next, global_norm=True)
                data.append((crop_curr, crop_next))
        self.data = data#np.array(data)
        print("Dataset ready!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_curr, img_next = self.data[idx]
        img_curr = Image.fromarray(np.uint8(img_curr * 255))
        img_next = Image.fromarray(np.uint8(img_next * 255))
        if self.transform:
            img_curr, img_next = self.transform(img_curr, img_next)
        return img_curr, img_next

    def inspect(self, out_dir="."):
        for sid, _ in enumerate(self.data[::self.__len__()//10]):
            sample = self.__getitem__(idx = sid)
            _, axs = plt.subplots(1, len(sample), figsize=(2,2))
            for im, ax in zip(sample, axs):
                ax.imshow(im[0], vmin=0, vmax=1, cmap="gray")
                ax.axis("off")
            plt.savefig(os.path.join(out_dir, f"sample_{sid:02d}"))

class DeltaDataset(Dataset):
    def __init__(self, reader, crop_size, transform=None):
        self.reader = reader
        self.crop_size = crop_size
        self.transform = transform
        self._generate_crops()

    def _generate_crops(self):
        data = []
        self._timepoint = []
        self._xcoord, self._ycoord = [], []
        self._mean_int, self._max_int = [], []
        nsamples_per_timepoint = 50
        s = self.crop_size
        T, C, Y, X = self.reader.shape
        for t in range(T-1):
            fov = self.reader[t:t+2].compute()
            msk = fov[0][-1].astype(int)
            msk[:, :s], msk[:, -s:], msk[:s, :], msk[-s:, :] = 0, 0, 0, 0
            msk = msk.astype(int)
            labels_available = np.unique(msk[msk>0].flatten())
            labels = np.random.choice(labels_available, size=nsamples_per_timepoint)
            for label in labels:
                yc, xc = [int(u.mean()) for u in np.where(msk==label)]
                yi = yc - int(s/2)
                yf = yi + s
                xi = xc - int(s/2)
                xf = xi + s
                for delta in [0, 1]:
                    crop = fov[delta][0][yi:yf, xi:xf]
                    self._timepoint.append(t+delta)
                    self._max_int.append(crop.max())
                    self._mean_int.append(crop.mean())
                    crop = proc.normalize_crop(crop, global_norm=False)
                    self._xcoord.append(xi)
                    self._ycoord.append(yi)
                    data.append(crop)
            self.data = np.array(data)
        print("Dataset ready!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = Image.fromarray(np.uint8(image * 255))  # Convert numpy array to PIL image
        if self.transform:
            image = self.transform(image)
        return image

    def get_crop_coordinates(self):
        return self._timepoint, self._xcoord, self._ycoord

    def get_crop_intensity_properties(self):
        return self._mean_int, self._max_int

class ZARRSignalCenteredDataset(Dataset):
    def __init__(self, reader, transform=None, debug=False):
        self.debug = debug
        self.reader = reader
        self.crop_size = const.M
        self.transform = transform
        self._generate_crops()

    def _generate_crops(self):
        data = []
        nsamples_per_timepoint = 50
        s = self.crop_size
        T, C, Y, X = self.reader.shape
        for t in range(1 if self.debug else T):
            fov = self.reader[t].compute()
            msk = (fov[0]>const.VMIN).astype(int)
            msk[:, :s], msk[:, -s:], msk[:s, :], msk[-s:, :] = 0, 0, 0, 0
            yy, xx = np.where(msk)
            idxs = np.random.choice(np.arange(len(xx)), size=nsamples_per_timepoint, replace=False)
            for idx in idxs:
                yc = yy[idx]
                xc = xx[idx]
                yi = yc - int(s/2)
                yf = yi + s
                xi = xc - int(s/2)
                xf = xi + s
                crop = fov[0][yi:yf, xi:xf]
                crop = proc.normalize_crop(crop, global_norm=False)
                data.append(crop)

        self.data = np.array(data)
        print("Dataset ready!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = Image.fromarray(np.uint8(image * 255))  # Convert numpy array to PIL image
        if self.transform:
            image = self.transform(image)
        return image

class ZARRSignalCenteredDatasetEval(Dataset):
    def __init__(self, reader, transform=None, debug=False):
        self.debug = debug
        self.reader = reader
        self.crop_size = const.M
        self.transform = transform
        self._generate_crops()

    def _generate_crops(self):
        data = []
        self._timepoint = []
        self._xcoord, self._ycoord = [], []
        self._mean_int, self._max_int = [], []
        nsamples_per_timepoint = 50
        s = self.crop_size
        T, C, Y, X = self.reader.shape
        for t in range(1 if self.debug else T):
            fov = self.reader[t].compute()
            msk = (fov[0]>const.VMIN).astype(int)
            msk[:, :s], msk[:, -s:], msk[:s, :], msk[-s:, :] = 0, 0, 0, 0
            yy, xx = np.where(msk)
            idxs = np.random.choice(np.arange(len(xx)), size=nsamples_per_timepoint, replace=False)
            for idx in idxs:
                yc = yy[idx]
                xc = xx[idx]
                yi = yc - int(s/2)
                yf = yi + s
                xi = xc - int(s/2)
                xf = xi + s
                crop = fov[0][yi:yf, xi:xf]
                self._max_int.append(crop.max())
                self._mean_int.append(crop.mean())
                self._timepoint.append(t)
                crop = proc.normalize_crop(crop, global_norm=True)
                self._xcoord.append(xi)
                self._ycoord.append(yi)
                data.append(crop)
        self.data = np.array(data)
        print("Dataset ready!")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = Image.fromarray(np.uint8(image * 255))  # Convert numpy array to PIL image
        if self.transform:
            image = self.transform(image)
        return image
    
    def get_crop_coordinates(self):
        return self._timepoint, self._xcoord, self._ycoord

    def get_crop_intensity_properties(self):
        return self._mean_int, self._max_int

class ZARRTemporalSignalCenteredDataset(Dataset):

    def __init__(self, reader, dts=[-2,-1,0,1], transform=None, debug=False):
        self._dts = dts
        self.debug = debug
        self.reader = reader
        self._eval_on = False
        self.crop_size = const.M
        self.transform = transform
        self._generate_crops()

    def _generate_crops(self):
        data = []
        self._timepoint = []
        self._xcoord, self._ycoord = [], []
        self._mean_int, self._max_int = [], []
        nsamples_per_timepoint = 200
        s = self.crop_size
        T, C, Y, X = self.reader.shape
        t0 = -np.min(self._dts)
        tf = np.max(self._dts)
        if t0 < 0:
            raise ValueError(f"Min dt must be negative. Got {t0}.")
        for t in [t0, T-tf-1] if self.debug else range(t0, T-tf):
            tps = [t+dt for dt in self._dts]
            fov = self.reader[tps].compute()
            msk = (fov[-2, 0]>const.VMIN).astype(int)
            msk[:, :s], msk[:, -s:], msk[:s, :], msk[-s:, :] = 0, 0, 0, 0
            yy, xx = np.where(msk)
            idxs = np.random.choice(np.arange(len(xx)), size=nsamples_per_timepoint, replace=False)
            for idx in idxs:
                yc = yy[idx]
                xc = xx[idx]
                yi = yc - int(s/2)
                yf = yi + s
                xi = xc - int(s/2)
                xf = xi + s
                self._xcoord.append(xi)
                self._ycoord.append(yi)
                self._timepoint.append(t)
                crop_curr = fov[:-1, 0, yi:yf, xi:xf]
                self._max_int.append(crop_curr.max())
                self._mean_int.append(crop_curr.mean())
                crop_curr = proc.normalize_crop(crop_curr, global_norm=False)
                crop_curr = np.moveaxis(crop_curr, 0, -1)
                crop_next = fov[-1, 0, yi:yf, xi:xf]
                crop_next = proc.normalize_crop(crop_next, global_norm=False)

                label1 = np.digitize(t, [0, 192, 384, 577]) - 1
                label2 = 0
                # reverse = np.random.rand()
                # if reverse < 0.333:
                #     label2 = 1
                #     temp = crop_curr[1].copy()
                #     crop_curr[1] = crop_curr[2].copy()
                #     crop_curr[2] = temp.copy()
                # elif reverse > 0.6666:
                #     label2 = 2
                #     crop_curr = crop_curr[::-1]

                data.append((crop_curr, crop_next, label1, label2))
        self.data = data
        print("Dataset ready!")

    def __len__(self):
        return len(self.data)

    def set_evaluation_mode_on(self):
        self._eval_on = True

    def __getitem__(self, idx):
        image_curr, image_next, label1, label2 = self.data[idx]
        image_curr = Image.fromarray(np.uint8(image_curr * 255))
        image_next = Image.fromarray(np.uint8(image_next * 255))
        if not self._eval_on:
            if self.transform:
                image_curr, image_next = self.transform(image_curr, image_next)
            return image_curr, image_next, label1, label2
        image_curr = self.transform(image_curr)
        return image_curr

    def get_crop_coordinates(self):
        return self._timepoint, self._xcoord, self._ycoord

    def get_crop_intensity_properties(self):
        return self._mean_int, self._max_int

    def inspect(self, out_dir="."):
        for sid in range(self.__len__()//10):
            img_curr, img_next = self.__getitem__(idx = sid)
            img_curr = np.moveaxis(np.array(img_curr), 0, -1)
            img_next = np.moveaxis(np.array(img_next), 0, -1)
            _, axs = plt.subplots(1, 2, figsize=(4,2))
            axs[0].imshow(img_curr, vmin=0, vmax=1)#, cmap="gray")
            axs[1].imshow(img_next, vmin=0, vmax=1)#, cmap="gray")
            axs[0].axis("off")
            axs[1].axis("off")
            plt.savefig(os.path.join(out_dir, f"sample_{sid:02d}"))
            import pdb; pdb.set_trace()

class ZARRTemporalSignalCenteredDatasetFromCoords(Dataset):

    def __init__(self, reader, dts=[-2,-1,0,1], coords=None, transform=None, debug=False):
        self._dts = dts
        self.debug = debug
        self.reader = reader
        self._eval_on = False
        self.crop_size = const.M
        self.transform = transform
        if coords is not None:
            self._generate_crops_from_coords(coords)
            return
        self._generate_crops()

    def _generate_crops(self):
        data = []
        self._timepoint = []
        self._xcoord, self._ycoord = [], []
        self._mean_int, self._max_int = [], []
        nsamples_per_timepoint = 200
        s = self.crop_size
        T, C, Y, X = self.reader.shape
        t0 = -np.min(self._dts)
        tf = np.max(self._dts)
        if t0 < 0:
            raise ValueError(f"Min dt must be negative. Got {t0}.")
        for t in [t0, T-tf-1] if self.debug else range(t0, T-tf):
            tps = [t+dt for dt in self._dts]
            fov = self.reader[tps].compute()
            msk = (fov[-2, 0]>const.VMIN).astype(int)
            msk[:, :s], msk[:, -s:], msk[:s, :], msk[-s:, :] = 0, 0, 0, 0
            yy, xx = np.where(msk)
            idxs = np.random.choice(np.arange(len(xx)), size=nsamples_per_timepoint, replace=False)
            for idx in idxs:
                yc = yy[idx]
                xc = xx[idx]
                yi = yc - int(s/2)
                yf = yi + s
                xi = xc - int(s/2)
                xf = xi + s
                self._xcoord.append(xi)
                self._ycoord.append(yi)
                self._timepoint.append(t)
                crop_curr = fov[:-1, 0, yi:yf, xi:xf]
                self._max_int.append(crop_curr.max())
                self._mean_int.append(crop_curr.mean())
                crop_curr = proc.normalize_crop(crop_curr, global_norm=False)
                crop_curr = np.moveaxis(crop_curr, 0, -1)
                crop_next = fov[-1, 0, yi:yf, xi:xf]
                crop_next = proc.normalize_crop(crop_next, global_norm=False)
                reverse = np.random.rand() < 0.5
                if reverse == 1:
                    temp = crop_curr[1].copy()
                    crop_curr[1] = crop_curr[2].copy()
                    crop_curr[2] = temp.copy()
                label1 = np.array([t<289], dtype=np.float32)
                label2 = np.array([reverse], dtype=np.float32)
                data.append((crop_curr, crop_next, label1, label2))
        self.data = data
        print("Dataset ready!")

    def _generate_crops_from_coords(self, coords):
        data = []
        t_last = -1
        self._timepoint = []
        self._xcoord, self._ycoord = [], []
        self._mean_int, self._max_int = [], []
        s = self.crop_size
        T, C, Y, X = self.reader.shape
        for (xc, yc, t) in coords:
            if t != t_last:
                tps = [t+dt for dt in self._dts]
                fov = self.reader[tps].compute()
            yi = yc - int(s/2)
            yf = yi + s
            xi = xc - int(s/2)
            xf = xi + s
            self._xcoord.append(xi)
            self._ycoord.append(yi)
            self._timepoint.append(t)
            # import pdb; pdb.set_trace()
            crop_curr = fov[:-1, 0, yi:yf, xi:xf]
            self._max_int.append(crop_curr.max())
            self._mean_int.append(crop_curr.mean())
            crop_curr = proc.normalize_crop(crop_curr, global_norm=False)
            crop_curr = np.moveaxis(crop_curr, 0, -1)
            crop_next = fov[-1, 0, yi:yf, xi:xf]
            crop_next = proc.normalize_crop(crop_next, global_norm=False)
            label1 = np.array([0], dtype=np.float32)
            label2 = np.array([0], dtype=np.float32)
            data.append((crop_curr, crop_next, label1, label2))
            t_last = t
        self.data = data
        print("Dataset ready!")

    def __len__(self):
        return len(self.data)

    def set_evaluation_mode_on(self):
        self._eval_on = True

    def __getitem__(self, idx):
        image_curr, image_next, label1, label2 = self.data[idx]
        image_curr = Image.fromarray(np.uint8(image_curr * 255))
        image_next = Image.fromarray(np.uint8(image_next * 255))
        if not self._eval_on:
            if self.transform:
                image_curr, image_next = self.transform(image_curr, image_next)
            return image_curr, image_next, label1, label2
        image_curr = self.transform(image_curr)
        return image_curr

    def get_crop_coordinates(self):
        return self._timepoint, self._xcoord, self._ycoord

    def get_crop_intensity_properties(self):
        return self._mean_int, self._max_int

    def inspect(self, out_dir="."):
        for sid in range(self.__len__()//10):
            img_curr, img_next = self.__getitem__(idx = sid)
            img_curr = np.moveaxis(np.array(img_curr), 0, -1)
            img_next = np.moveaxis(np.array(img_next), 0, -1)
            _, axs = plt.subplots(1, 2, figsize=(4,2))
            axs[0].imshow(img_curr, vmin=0, vmax=1)#, cmap="gray")
            axs[1].imshow(img_next, vmin=0, vmax=1)#, cmap="gray")
            axs[0].axis("off")
            axs[1].axis("off")
            plt.savefig(os.path.join(out_dir, f"sample_{sid:02d}"))
            import pdb; pdb.set_trace()

class ZARRTemporalRandom(Dataset):

    def __init__(self, reader, dts=[-2,-1,0,1], transform=None, debug=False):
        self._dts = dts
        self.debug = debug
        self.reader = reader
        self._eval_on = False
        self.crop_size = const.M
        self.transform = transform
        self._generate_crops()

    def _generate_crops(self):
        data = []
        self._timepoint = []
        self._xcoord, self._ycoord = [], []
        self._mean_int, self._max_int = [], []
        nsamples_per_timepoint = 200
        s = self.crop_size
        T, C, Y, X = self.reader.shape
        t0 = -np.min(self._dts)
        tf = np.max(self._dts)
        if t0 < 0:
            raise ValueError(f"Min dt must be negative. Got {t0}.")
        for t in [t0, T-tf-1] if self.debug else range(t0, T-tf):
            tps = [t+dt for dt in self._dts]
            fov = self.reader[tps].compute()
            idxs = np.arange(nsamples_per_timepoint)
            xx = np.random.randint(const.M, fov.shape[-1]-const.M, size=nsamples_per_timepoint)
            yy = np.random.randint(const.M, fov.shape[-2]-const.M, size=nsamples_per_timepoint)
            for idx in idxs:
                yc = yy[idx]
                xc = xx[idx]
                yi = yc - int(s/2)
                yf = yi + s
                xi = xc - int(s/2)
                xf = xi + s
                self._xcoord.append(xi)
                self._ycoord.append(yi)
                self._timepoint.append(t)
                crop_curr = fov[:-1, 0, yi:yf, xi:xf]
                self._max_int.append(crop_curr.max())
                self._mean_int.append(crop_curr.mean())
                crop_curr = proc.normalize_crop(crop_curr, global_norm=False)
                crop_curr = np.moveaxis(crop_curr, 0, -1)
                crop_next = fov[-1, 0, yi:yf, xi:xf]
                crop_next = proc.normalize_crop(crop_next, global_norm=False)

                label1 = np.digitize(t, [0, 192, 384, 577]) - 1
                label2 = 0

                data.append((crop_curr, crop_next, label1, label2))
        self.data = data
        print("Dataset ready!")

    def __len__(self):
        return len(self.data)

    def set_evaluation_mode_on(self):
        self._eval_on = True

    def __getitem__(self, idx):
        image_curr, image_next, label1, label2 = self.data[idx]
        image_curr = Image.fromarray(np.uint8(image_curr * 255))
        image_next = Image.fromarray(np.uint8(image_next * 255))
        if not self._eval_on:
            if self.transform:
                image_curr, image_next = self.transform(image_curr, image_next)
            return image_curr, image_next, label1, label2
        image_curr = self.transform(image_curr)
        return image_curr

    def get_crop_coordinates(self):
        return self._timepoint, self._xcoord, self._ycoord

    def get_crop_intensity_properties(self):
        return self._mean_int, self._max_int

    def inspect(self, out_dir="."):
        for sid in range(self.__len__()//10):
            img_curr, img_next = self.__getitem__(idx = sid)
            img_curr = np.moveaxis(np.array(img_curr), 0, -1)
            img_next = np.moveaxis(np.array(img_next), 0, -1)
            _, axs = plt.subplots(1, 2, figsize=(4,2))
            axs[0].imshow(img_curr, vmin=0, vmax=1)#, cmap="gray")
            axs[1].imshow(img_next, vmin=0, vmax=1)#, cmap="gray")
            axs[0].axis("off")
            axs[1].axis("off")
            plt.savefig(os.path.join(out_dir, f"sample_{sid:02d}"))
            import pdb; pdb.set_trace()
