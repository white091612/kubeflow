import os
from collections import defaultdict
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from lion_pytorch import Lion
from torchsampler import ImbalancedDatasetSampler
from PIL import Image

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class Transforms:
    def __init__(self, img_size, mean_std, data='train'):
        self.transforms = self.image_transform(img_size, mean_std, data)
    def image_transform(self, img_size, mean_std, data):
        if data == 'train':
            return A.Compose(
                [
                    A.RandomCrop(224, 224),
                    A.RandomBrightnessContrast(),
                    A.ImageCompression(quality_lower=85, quality_upper=100, p=0.5),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                    A.RandomContrast(limit=0.2, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                    A.Cutout(always_apply=False, p=0.5, num_holes=10, max_h_size=10, max_w_size=10),
                    A.CLAHE(always_apply=False, p=0.5, clip_limit=(1, 15), tile_grid_size=(8, 8)),
                    A.Equalize(always_apply=False, p=0.5, mode='cv', by_channels=False),
                    A.GaussNoise(always_apply=False, p=0.5, var_limit=(0.0, 26.849998474121094)),
                    A.Normalize((mean_std[0], mean_std[1], mean_std[2]), (mean_std[3], mean_std[4], mean_std[5])),
                    ToTensorV2()
                ]
            )
        elif data == 'val':
            return A.Compose(
                [
                    A.CenterCrop(img_size, img_size),
                    A.Normalize((mean_std[0], mean_std[1], mean_std[2]), (mean_std[3], mean_std[4], mean_std[5])),
                    ToTensorV2()
                ]
            )
        # (TTA)Test Time Augmentation
        elif data == 'test':
            return A.Compose(
                [
                    A.CenterCrop(img_size, img_size),
                    # A.Normalize((mean_std[0], mean_std[1], mean_std[2]), (mean_std[3], mean_std[4], mean_std[5])),
                    ToTensorV2()
                ]
            )

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))['image']
class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()
    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})
    def update(self, metric_name, val):
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]
    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
def build_dataset(data_path, img_size, batch_size, mean_std):
    train_dataset = CustomImageFolder(f'{data_path}/dataset/train', transform=Transforms(img_size, mean_std, 'train'))
    val_dataset = CustomImageFolder(f'{data_path}/dataset/val', transform=Transforms(img_size, mean_std, 'val'))
    test_dataset = CustomImageFolder(f'{data_path}/dataset/test', transform=Transforms(img_size, mean_std, 'test'))
    train_loader = DataLoader(train_dataset, batch_size,
                              sampler=ImbalancedDatasetSampler(train_dataset),
                              num_workers=0, pin_memory=True, shuffle=False, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size,
                            num_workers=0, pin_memory=True, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size,
                             num_workers=0, pin_memory=True, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader
def build_optimizer(model, optimizer, learning_rate):
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate)
    elif optimizer == "lion":
        optimizer = Lion(model.parameters(),
                         lr=learning_rate, weight_decay=1e-2)
    return optimizer


class CustomImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = sorted(
            [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d != '.ipynb_checkpoints'])
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.img_paths = self.get_img_paths()

    def get_img_paths(self):
        img_paths = []
        for target_class in self.classes:
            class_dir = os.path.join(self.root, target_class)
            for img_name in os.listdir(class_dir):
                if not img_name.startswith('.') and os.path.splitext(img_name)[-1] in IMG_EXTENSIONS:
                    img_path = os.path.join(class_dir, img_name)
                    img_paths.append((img_path, self.class_to_idx[target_class]))
        return img_paths

    def get_labels(self, *args):
        return [x[1] for x in self.img_paths]

    def __getitem__(self, index):
        img_path, target = self.img_paths[index]
        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.img_paths)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')