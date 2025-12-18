from predictor import PredictionService
from PIL import Image
import torch
from utils.model import load_trained_model, model_to_tta_model
import os

prefix = r"H:\From F\springhouse-svc-pytorch-classifier\src\scripts\local_docker\test_dir"
model_path = "/home/travis/springhouse-svc-pytorch-classifier/artifacts/best-model:v0/model.pt"
img_size = 384
class_count = 40
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_path = r"12-0_0-13-20200310072621.jpg"

labels_path = os.path.join(prefix, "model", "labels.txt")
try:
    with open(labels_path) as f:
        labels = f.read().splitlines()
except:
    print("labels file not found, will not return class name")
    labels = []

model = load_trained_model(model_path, device, class_count)


img = Image.open(image_path)

out = PredictionService.predict(model, img, labels)
print(out)

# model = model_to_tta_model(model)
# out = PredictionService.predict(model, img, labels)
# print(out)