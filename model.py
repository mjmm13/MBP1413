import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_model(num_classes=2, pretrained=True):
    # load pretrained mask rcnn with optional pretrained weights
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)

    # get number of input features for the classifier
    inp = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one 
    model.roi_heads.box_predictor = FastRCNNPredictor(inp, num_classes)

    # now get the number of input features for the mask classifier
    in_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask with our own
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features,
                                                       hidden_layer,
                                                       num_classes)

    return model

