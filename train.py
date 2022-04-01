import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import KaggleDataset, collate
from model import get_model
from test import calc_accuracy
   
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
N_EPOCHS = 5 
model = get_model().to(DEVICE)
model.eval()

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


train_dataset = KaggleDataset("/home/mmcneil/amartel_data3/mmcneil/kaggle_2018/stage1_train/")
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate)

val_dataset = KaggleDataset("/home/mmcneil/amartel_data3/mmcneil/kaggle_2018/stage1_test/", transforms=False)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate)

for epoch in range(N_EPOCHS):
    loss_totals = {"loss_classifier":0, "loss_box_reg":0, "loss_mask":0,
                   "loss_objectness":0, "loss_rpn_box_reg":0}
    for images, labels in tqdm(train_dataloader):
        model.train()
        optimizer.zero_grad()
        images = list(image.to(DEVICE) for image in images)
        # move label dict to DEVICE, meanwhile check you haven't transformed labels away
        # If you have skip this batch
        new = []
        no_label = False
        for t in labels:
            new_dict = {}
            for k, v in t.items():
                if not v.shape[0]:
                    no_label = True
                    continue
                else:
                    new_dict[k] = v.to(DEVICE)
            new.append(new_dict)
        if no_label:
            continue
        labels = new
        losses = model(images, labels)
        loss = sum(l for l in losses.values())
        loss.backward()
        optimizer.step()
        loss_totals = {k: v + loss_totals[k] for k, v in losses.items()}
        
    lr_scheduler.step()
    print(loss_total)
        
    accuracy = 0
    precision = 0
    recall = 0
    for images, labels in tqdm(val_dataloader):
        model.eval()
        images = list(image.to(DEVICE) for image in images)
        labels = [{k: v.to(DEVICE) for k, v in l.items()} for l in labels]
        res = model(images)
        acc, pres, rec = calc_accuracy(res, labels)

        accuracy += acc
        precision += pres
        recall += rec
    
    number = len(val_dataloader)   
    print("Accuracy: {:.6f}, Precision: {:.6f}, Recall: {:.6f}".format(
          accuracy/number, precision/number, recall/num))
    
torch.save(model.state_dict(), "10.pt")

