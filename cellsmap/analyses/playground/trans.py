import torch
import torchvision.transforms as transforms

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

class AddZeroNoise(object):
    def __init__(self, probability=0.1):
        self.probability = probability  # Probability of zeroing a pixel

    def __call__(self, tensor):
        mask_shape = tensor.shape[1:]
        mask = torch.rand(mask_shape) > self.probability
        mask = mask.unsqueeze(0)  # Add a channel dimension at the beginning
        return tensor * mask

class MainTransform:
    def __init__(self, add_noise=False, probability=0.1):
        self.transforms = [
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, scale=(0.9, 1.1)),
            transforms.ToTensor()
        ]
        if add_noise:
            self.transforms.append(AddZeroNoise(probability=probability))
        self.transform_pipeline = transforms.Compose(self.transforms)
    def __call__(self, img):
        return self.transform_pipeline(img)

class PairedTransform():
    def __init__(self, add_noise=True):
        self.transform1 = MainTransform(add_noise=add_noise, probability=0.5)
        self.transform2 = MainTransform()
    
    def __call__(self, img_curr, img_next):
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
        torch.manual_seed(seed)
        img_curr = self.transform1(img_curr)
        torch.manual_seed(seed)
        img_next = self.transform2(img_next)
        return img_curr, img_next