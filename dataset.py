import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torchvision.transforms.functional as TF

MAX_MAGNITUDE = 100
PYTORCH_TRANSFORMS = [(ImageOps.autocontrast, 0),
                      (ImageOps.solarize, torch.linspace(255.0, 0.0, MAX_MAGNITUDE)),
                      (ImageOps.posterize, 8 - (torch.arange(MAX_MAGNITUDE) / ((MAX_MAGNITUDE - 1) / 4)).round().int()),
                      (TF.adjust_contrast, torch.linspace(0.0, 0.9, MAX_MAGNITUDE)),
                      (TF.adjust_brightness, torch.linspace(0.0, 0.9, MAX_MAGNITUDE)),
                      (TF.rotate, torch.linspace(0.0, 180.0, MAX_MAGNITUDE)),
                      (TF.affine, torch.linspace(0.0, 1, MAX_MAGNITUDE)),
                      (TF.affine, torch.linspace(0.0, 1, MAX_MAGNITUDE)),
                      (TF.affine, torch.linspace(0.0, 0.5, MAX_MAGNITUDE)),
                      (TF.affine, torch.linspace(0.0, 0.5, MAX_MAGNITUDE))]

def collate(data):
    images, targets = [], []
    for i in range(len(data)):
        images.append(data[i][0])
        targets.append(data[i][1])
    return images, targets
                
            
class KaggleDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=True, K=2, M=10):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.folders = list(sorted(os.listdir(root)))
        self.K = K
        self.M = M
        
    def get_transforms(self):
        K_transforms = random.choices(range(len(PYTORCH_TRANSFORMS)), k=self.K)
        magnitude = random.choice(range(self.M))
        
        return K_transforms, magnitude
        
    def perform_transforms(self, img, K_transforms=None, magnitude=None, mask=False):
        if not self.transforms:
            return TF.to_tensor(img)
        else:
            for index in K_transforms:
                transformation, mag = PYTORCH_TRANSFORMS[index]
                if index == 0 and not mask:
                    img = transformation(img)
                elif index < 3 and not mask:
                    img = transformation(img, mag.tolist()[magnitude])
                elif index < 5 and not mask:
                    img = transformation(img, mag[magnitude])
                elif index == 5:
                    img = transformation(img, mag[magnitude])
                elif index > 5:
                    x_y = index % 2
                    if index < 7:
                        shear = [0,0]
                        shear[x_y] = mag[magnitude] * 180
                        img = transformation(img, shear=shear, angle=0, translate=(0,0), scale=1)
                    else:
                        trans = [0,0]
                        trans[x_y] = mag[magnitude] * img.size[x_y]
                        img = transformation(img, shear=0, angle=0, translate=tuple(trans), scale=1)
                    
        return TF.to_tensor(img)

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, self.folders[idx], "images")
        mask_path = os.path.join(self.root, self.folders[idx], "masks")
        img_name = os.listdir(img_path)[0]
        img = Image.open(os.path.join(img_path, img_name)).convert("RGB")
        new_size = img.size
        # Create a list for transforms of this index
        if self.transforms == True:
            K_transforms, magnitude = self.get_transforms()
        else:
            K_transforms, magnitude = None, None
        # Perform transforms on the original image
        img = self.perform_transforms(img, K_transforms, magnitude)
        
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        boxes = torch.Tensor([]).view((0,4)).type(torch.float)
        labels = torch.Tensor([]).type(torch.int64)
        masks = torch.Tensor([]).view(0, new_size[1], new_size[0]).type(torch.uint8)
        for mask_name in os.listdir(mask_path):
            if mask_name.split(".")[1] != "png":
                continue
            mask_raw = Image.open(os.path.join(mask_path, mask_name)).resize(new_size)
            #data = np.asarray(mask_raw)
            data = self.perform_transforms(mask_raw, K_transforms, magnitude, mask=True)
            pos = np.where(data)
            if not np.any(pos[1]):
                # Nucleus at this location is presumed to be missing due to the transforms
                continue
            x_min = np.min(pos[2])
            x_max = np.max(pos[2])
            y_min = np.min(pos[1])
            y_max = np.max(pos[1])
            if x_min == x_max:
                x_max += 1
            if y_min == y_max:
                y_max += 1
            #img1 = ImageDraw.Draw(img)
            #img1.rectangle([(x_min,y_min),(x_max,y_max)], outline="red")
            #img.save(self.folders[idx] + ".png")
            data = torch.Tensor(data)
            bound_box = torch.Tensor([x_min, y_min, x_max, y_max]).unsqueeze(0)
            boxes = torch.cat((boxes, bound_box.type(torch.float)))
            labels = torch.cat((labels, torch.Tensor([1]).type(torch.int64)))
            masks = torch.cat((masks, data.type(torch.uint8)))

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.Tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        return len(self.folders)

