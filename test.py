import os

import numpy as np
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
    
def calc_accuracy(res, labels, thres=0.5):
    accuracy = 0
    precision = 0
    recall = 0
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
        correct = int(torch.sum(best_IOU > thres))
        
        precision += correct / num_called
        recall += correct / num_nuclei
        
        accuracy += correct / (num_nuclei + num_called - correct)
        
    return accuracy, precision, recall
    
def main():
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
        for images, labels in tqdm(test_dataloader):
            images = list(image.to(DEVICE) for image in images)
            labels = [{k: v.to(DEVICE) for k, v in l.items()} for l in labels]
            res = model(images)
            del images
            acc, pres, rec = calc_accuracy(res, labels)

            accuracy += acc
            precision += pres
            recall += rec

    print("Accuracy: {:.6f}, Precision: {:.6f}, Recall: {:.6f}".format(
           accuracy/106, precision/106, recall/106))
