from utils.data import get_data_loaders

import json
import os
import torch
from utils.model import load_trained_model

prefix = os.path.join("src", "scripts", "local_docker", "test_dir")
param_path = os.path.join(prefix, "input", "config", "hyperparameters.json")

######################################################
model_path = r"C:\Users\Admin\Downloads\11-2\model_1_.pt"
img_size = 384
batch_size = 8
limit = 20
crop = True
train_augmentation = True
train_path = r"H:\2022-10-05-wk1-4-plus\input\data\train_xval"
######################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = load_trained_model(model_path, device, 725)

    # Read in any hyperparameters that the user passed with the training job
with open(param_path, 'r') as tc:
    trainingParams = json.load(tc)

trainloader, valloader, testloader = get_data_loaders(train_path, img_size, batch_size=batch_size, train_augmentation=train_augmentation, crop=crop)


running_loss = 0.0    
correct = 0
total = 0
criterion = torch.nn.CrossEntropyLoss()

test_counter = 0

for data in valloader:
        test_counter += 1
        if test_counter > limit:
            break
        images, labels = data
        labels = labels.to(device)

        # calculate outputs by running images through the network
        outputs = model(images.to(device))

        loss = criterion(outputs, labels.to(device))



    # print statistics
        running_loss += loss.item()
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()




accuracy = 100 * correct // total
print(f'Validation accuracy of the network: {accuracy} %')
print("val loss", str(running_loss / len(valloader)))