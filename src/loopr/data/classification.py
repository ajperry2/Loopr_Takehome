from zlib import crc32
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import defaultdict
import numpy as np
import cv2
import torch
import random

from loopr.config.training_nn import TrainingNNConfig


class ClassificationDataset(Dataset):
    def __init__(
        self, 
        data_dir: Path = TrainingNNConfig.data_dir, 
        transforms=None, 
        training=True, 
        train_split = 0.8, 
        censor_files = False, 
        limited_classes = dict(), crop = True
    ):
        defect_image_dir = data_dir / "Defect_images"
        mask_image_dir = data_dir / "Mask_images"
        no_defect_image_dir = data_dir / "NODefect_images"
        self.file_to_mask_file = lambda image_file: list(self.mask_dir.glob(f"*_{image_file.name.split("_")[1].zfill(3)}_*"))[0]
        self.file_has_mask_file = lambda image_file: len(list(self.mask_dir.glob(f"*_{image_file.name.split("_")[1].zfill(3)}_*")))>0
        
        self.defect_dir = defect_image_dir
        self.all_classes = TrainingNNConfig.kept_classes+TrainingNNConfig.unkept_classes
        self.no_defect_dir = no_defect_image_dir
        self.mask_dir = mask_image_dir
        all_files = [path for path in self.defect_dir.glob("**/*.png") ]
        all_files = [file for file in all_files if self.file_has_mask_file(file)]
        self.all_is_file_defect = [True for i in range( len(all_files))]
        all_files += [path for path in self.no_defect_dir.glob("**/*.png")]
        self.crop = crop
        
        self.all_is_file_defect += [False for path in self.no_defect_dir.glob("**/*.png")]
        
        self.files = []
        self.is_file_defect = []
        self.classes = []
        counts = defaultdict(int)
        for i, file in enumerate(all_files):
            
            hash_value = float(crc32(file.name.encode()) & 0xffffffff) / 2**32
            class_label = int(file.name.split("_")[1])
            if self.all_is_file_defect[i] and class_label not in TrainingNNConfig.kept_classes:
                hash_value = 1.0
                
            if len(limited_classes)>0 and class_label in limited_classes : 

                if class_label in self.all_classes:
                    if counts[self.all_classes.index(class_label)+1] > limited_classes[class_label]:
                        continue
                else:
                    if counts[0] > limited_classes[0]:
                        continue
            part_of_dataset = (
                (hash_value < train_split and training) 
                or (hash_value >= train_split and not training)
            )
            image_id = int(file.name.split("_")[0])
            if part_of_dataset:
                class_label = int(file.name.split("_")[1])

                if class_label in TrainingNNConfig.kept_classes or not self.all_is_file_defect[i] or not training:
                    if self.all_is_file_defect[i]:
                        if class_label not in TrainingNNConfig.kept_classes and censor_files:
                            continue
                        self.classes.append(self.all_classes.index(class_label)+1 ) 
                        self.files.append(file)
                        counts[self.all_classes.index(class_label)+1] += 1
                    else:
                        self.files.append(file)
                        self.classes.append(0)
                        counts[0] += 1
        self.transforms = transforms

        self.all_is_file_defect = [True for i in range( len(all_files))]

    def __len__(self):
        return len(self.files)

    def _read_image(self, image_file):
        img = cv2.imread(str(image_file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(image_id)
        return img
            
    def __getitem__(self, idx):
        image_file = self.files[idx]
        img = self._read_image(image_file)
        img = np.repeat(img[..., None], 3, axis=2)
        
        # img = torch.from_numpy(img).float() / 255.0
        if self.transforms:
            
            out = self.transforms(image=img)
            img= out["image"]  # -> (C,H,W)
        else:
            img = torch.from_numpy(img.transpose(2,0,1)).float()

        # Crop around mask
        sample_choice = random.uniform(0, 1) < 0.7
        if self.crop:
            if self.file_has_mask_file(image_file) and sample_choice:
                mask = cv2.imread(str(self.file_to_mask_file(image_file)), cv2.IMREAD_GRAYSCALE)
    
                mask = mask != 0
                yx = torch.from_numpy(np.argwhere(mask))
    
                y = yx[:,0]
                x = yx[:,1]
                x_cen = x.float().mean()
                x_cen += int(np.random.randn() * TrainingNNConfig.width/2)
                
                x_start = int(min(max(0, x_cen - TrainingNNConfig.width/2), img.shape[2]-TrainingNNConfig.width))
    
                x_end = x_start + TrainingNNConfig.width
    
                img = img[:,:,x_start:x_end]
            else:
    
                x_start = int(np.random.rand() * (img.shape[2]-TrainingNNConfig.width))
                x_end = int(x_start + TrainingNNConfig.width)
                img = img[:,:,x_start:x_end]
                # print(2, img.shape)
                
            
            
        
        class_ = self.classes[idx]
        meta = {"idx": idx}
        return img, class_, meta


# --- Collate function ---
def collate_fn(batch):
    images, classes, metas = zip(*batch)
    images = torch.stack(images)
    classes = torch.stack(classes)
    return images, classes, metas


def get_train_transforms(
    H=TrainingNNConfig.height, 
    W=TrainingNNConfig.width
):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9,1.1), translate_percent=(0.01,0.01), rotate=(-1,1), p=0.5),
        A.GaussianBlur( blur_limit=(3, 7), p=0.1),
        A.RandomBrightnessContrast(p=0.4),
        # A.RandomCrop(
        #     H,
        #     W,
        # ),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ], is_check_shapes=False)

def get_valid_transforms(
    H=TrainingNNConfig.height, 
    W=TrainingNNConfig.width
):
    return A.Compose([
        # A.RandomCrop(H, W, pad_if_needed=True),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])