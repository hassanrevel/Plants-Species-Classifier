import os
import json
import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(VisionTransformer, self).__init__()
        self.backbone = models.vit_b_16(pretrained=pretrained)  # Load pretrained ViT backbone
        self.backbone.heads = nn.Linear(self.backbone.heads.head.in_features, num_classes)  # Change output to num_classes

    def forward(self, x):
        x = self.backbone(x)
        return x

def model_fn(model_dir):
    model = VisionTransformer(11, pretrained=False)
    model.load_state_dict(
        torch.load(os.path.join(model_dir,  "model_0.1_best.pth"),
                   map_location=device))
    model.to(device).eval()
    return model

def input_fn(request_body, request_content_type):
    assert request_content_type=='application/json'
    data = json.loads(request_body)['inputs']
    data = torch.tensor(data, dtype=torch.float32, device=device)
    return data

def predict_fn(input_object, model):
    with torch.no_grad():
        prediction = model(input_object)
        _, predicted = torch.max(prediction, 1)
    return predicted

def output_fn(predictions, content_type):
    assert content_type == 'application/json'
    res = predictions.cpu().numpy().tolist()
    return json.dumps(res)