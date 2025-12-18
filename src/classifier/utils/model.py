import torch
import torch.nn as nn
import torch.nn.functional as F
import ttach as tta

from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

def load_trained_model(path, device, class_count):
    weights = torch.load(path)
    model = Net(class_count, load_weights=False)
    model.load_state_dict(weights, strict=False)
    model.to(device)
    model.eval()
    return model

def model_to_tta_model(model):
    tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.ten_crop_transform(350, 350))
    return tta_model

class Net(nn.Module):
    def __init__(self, class_count=15, load_weights=True):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT

        if load_weights:
            self.backbone = efficientnet_v2_s(weights=weights)
        else:
            self.backbone = efficientnet_v2_s()
        self.backbone.classifier = nn.Identity()
        self.fc = nn.Linear(1280, class_count)

        self.preprocess = weights.transforms()

    def forward(self, x):
        x = self.preprocess(x)
        x = self.backbone(x)
        x = self.fc(x)
        return x
