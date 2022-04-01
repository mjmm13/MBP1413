import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from tqdm import tqdm

from dataset import KaggleDataset, collate
from model import get_model

MODEL_PATH = "10.pt"
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
def check_example(images, res, labels, counter, good_thres=0.9, bad_thres=0.2):
    for i, instance in enumerate(res):
        gen_masks = instance["masks"]
        true_masks = labels[i]["masks"].unsqueeze(0)
        fil = instance["scores"] > 0.5
        gen_masks = gen_masks[fil]
        num_nuclei = true_masks.shape[1]
        num_called = gen_masks.shape[0]
        if num_nuclei > 100 or num_called > 100:
            gen_masks, true_masks = gen_masks.cpu(), true_masks.cpu()
        intersection = torch.sum(true_masks * gen_masks, dim=(2,3))
        union = torch.max(gen_masks, true_masks.type(torch.float))
        union = torch.sum(union, dim=(2,3))
        
        IOU = intersection / union
        best_IOU = torch.max(IOU, dim=1).values
        correct = int(torch.sum(best_IOU > 0.5))
        
        accuracy = correct / (num_nuclei + num_called - correct)
        if accuracy > good_thres:
            out_mask = torch.sum(gen_masks, dim=(0,1))
            img = Image.fromarray(np.uint8(out_mask.cpu().numpy()*255))
            img.save(str(counter) + "_generated_mask.png")
            out_mask = torch.sum(true_masks, dim=(0,1))
            img = Image.fromarray(np.uint8(out_mask.cpu().numpy()*255))
            img.save(str(counter) + "_real_mask.png")
            img = Image.fromarray(np.uint8(images[i].cpu().squeeze(0).numpy().transpose(1,2,0)*255))
            img.save(str(counter) + "_real.png")
            
        if accuracy < bad_thres:
            out_mask = torch.sum(gen_masks, dim=(0,1))
            img = Image.fromarray(np.uint8(out_mask.cpu().numpy()*255))
            img.save(str(counter) + "_generated_mask_bad.png")
            out_mask = torch.sum(true_masks, dim=(0,1))
            img = Image.fromarray(np.uint8(out_mask.cpu().numpy()*255))
            img.save(str(counter) + "_real_mask_bad.png")
            img = Image.fromarray(np.uint8(images[i].cpu().squeeze(0).numpy().transpose(1,2,0)*255))
            img.save(str(counter) + "_real_bad.png")
        
    return None
    

model = get_model()
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)
model.eval()

test_dataset = KaggleDataset("/home/mmcneil/amartel_data3/mmcneil/kaggle_2018/stage2_test/", transforms=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate)

accuracy = 0
precision = 0
recall = 0
    
with torch.no_grad():
    counter = 0
    for images, labels in tqdm(test_dataloader):
        images = list(image.to(DEVICE) for image in images)
        labels = [{k: v.to(DEVICE) for k, v in l.items()} for l in labels]
        res = model(images)
        check_example(images, res, labels, counter)

        counter += 1

