from zlib import crc32
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
import numpy as np
import cv2
import torch

from loopr.config.training_nn import TrainingNNConfig


class SemanticSegmentationDataset(Dataset):
    def __init__(self, transforms=None, training=True, val_split=TrainingNNConfig.train_split, load_rgb=True, censor_class: dict = dict()):
        data_dir = TrainingNNConfig.data_dir
        defect_image_dir = data_dir / "Defect_images"
        mask_image_dir = data_dir / "Mask_images"
        no_defect_image_dir = data_dir / "NODefect_images"
        self.all_classes = ["none"]+TrainingNNConfig.kept_classes
        self.defect_dir = defect_image_dir
        self.no_defect_dir = no_defect_image_dir
        self.mask_dir = mask_image_dir
        self.transforms = transforms
        self.num_classes = len(self.all_classes)
        self.load_rgb = load_rgb
        
        all_files = [path for path in self.defect_dir.glob("**/*.png") if int(path.name.split("_")[1]) in TrainingNNConfig.kept_classes]
        self.files= all_files
        
        # Make sure all files have masks
        self.file_has_mask_file = lambda image_file: len(list(self.mask_dir.glob(f"*_{image_file.name.split("_")[1].zfill(3)}_*")))>0
        self.file_to_mask_file = lambda image_file: list(self.mask_dir.glob(f"*_{image_file.name.split("_")[1].zfill(3)}_*"))[0]
        self.files = [file for file in self.files if self.file_has_mask_file(file)]


        self.file_is_kept_class = lambda image_file: int(image_file.name.split("_")[1]) in TrainingNNConfig.kept_classes
        self.files = [file for file in self.files if self.file_is_kept_class(file)]
        
        
        # only train/val
        self.part_of_dataset = lambda p: (
            (float(crc32(p.name.encode()) & 0xffffffff) / 2**32 > 1-val_split and not training) 
            or (float(crc32(p.name.encode()) & 0xffffffff) / 2**32 <= 1-val_split and training)
        )
        self.files = [file for file in self.files if self.part_of_dataset(file)]

        self.classes = [
            self.all_classes.index(int(path.name.split("_")[1]))
            for path
            in self.files
        ]


        self.files += [path for path in self.no_defect_dir.glob("**/*.png")]
        self.classes += [
            0
            for path
            in [path for path in self.no_defect_dir.glob("**/*.png")]
        ]
        
        from collections import Counter
        class_counts = Counter(self.classes)
        for class_index, max_count in censor_class.items():
            to_removes = []
            curr_count = 0 
            for c, path in zip(self.classes, self.files):
                to_remove = False
                if c == class_index:
                    curr_count += 1
                    if curr_count >= max_count:
                        to_remove = True
                to_removes.append(to_remove)
            self.classes = [c for i,c in enumerate(self.classes) if not to_removes[i]]
            self.files = [f for i,f in enumerate(self.files) if not to_removes[i]]
        
    def __len__(self):
        return len(self.files)

    def _read_image(self, image_file):
        img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(image_file)
        if self.load_rgb:
            img = np.repeat(img[..., None], 3, axis=2)  # (H,W,3)
        return img

    def _build_mask(self, index):
        image_file = self.files[index]
        class_ = self.classes[index]
        img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        shape = img.shape

        
        is_defect = class_ != 0
        m = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
        
        if is_defect:

            mask_file = self.file_to_mask_file(image_file)
            mask_img = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            mask_img[mask_img!=0] = 1
            m[:,:,class_] = mask_img
            return m


        
        else:
            return m
            
    def __getitem__(self, idx):
        image_file = self.files[idx]
        img = self._read_image(image_file)
        mask = self._build_mask(idx)  # (H,W,C)

        if self.transforms:
            out = self.transforms(image=img, mask=mask)
            img, mask = out["image"], out["mask"].permute(2,0,1)  # -> (C,H,W)
        else:
            img = torch.from_numpy(img.transpose(2,0,1)).float()
            mask = torch.from_numpy(mask.transpose(2,0,1)).float()

        if self.file_has_mask_file(image_file):
            mask = cv2.imread(str(self.file_to_mask_file(image_file)), cv2.IMREAD_GRAYSCALE)
            mask = mask != 0
            x, y = np.where(mask)
            x_cen = x.mean()
            x_cen += int(np.random.randn() * TrainingNNConfig.width/2)
            
            x_start = min(max(0, x_cen - TrainingNNConfig.width/2), img.shape[2]-TrainingNNConfig.width)
            x_end = x_start + TrainingNNConfig.width
            img = img[:,:,x_start:x_end]
        else:
            x, y = np.where(mask)
            x_start = np.random.randn() * (img.shape[2]-TrainingNNConfig.width)
            x_end = x_start + TrainingNNConfig.width
            img = img[:,:,x_start:x_end]
            
        
        meta = {"image_id": idx}
        return img, mask, meta

# --- Collate function ---
def collate_fn(batch):
    images, classes, metas = zip(*batch)
    images = torch.stack(images)
    classes = torch.stack(classes)
    return images, classes, metas
    
def get_train_transforms(H=TrainingNNConfig.height, W=TrainingNNConfig.width):
    return A.Compose([
        # A.HorizontalFlip(p=0.5),
        # A.Affine(scale=(0.9,1.1), translate_percent=(0.02,0.02), rotate=(-1,1), p=0.5),
        # A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.RandomBrightnessContrast(p=0.4),
        A.RandomCrop(
            H,
            W,
        ),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ], is_check_shapes=False)

def get_valid_transforms(H=TrainingNNConfig.height, W=TrainingNNConfig.width):
    return A.Compose([
        A.RandomCrop(H, W, pad_if_needed=True),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])
