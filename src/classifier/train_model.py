from tqdm import tqdm
import os
import numpy as np
import json

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support 
import matplotlib.pyplot as plt

from utils.autobatch import check_train_batch_size
from utils.data import get_data_loaders, save_model_input_to_disk
from utils.model import Net

import wandb

def get_class_names(train_path):
    result = []
    json_path = train_path + os.sep + "data_config.yaml"
    try:
        with open(json_path) as f:
            data = json.load(f)
        return data["names"]
    except:
        return []

    
    idx_to_name = {}
    max_idx = -1
    for food, idx in data.items():
        idx_int = int(idx)
        idx_to_name[idx] = food
        max_idx = max(max_idx, idx_int)
    
    if max_idx == -1:
        return ["item"]
    
    for i in range(max_idx + 1):
        result.append(idx_to_name[i])

    return result

def train_main(**kwargs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #==========================
    run_name = kwargs["run_name"]
    run_path = "/opt/ml/output" + os.sep + run_name
    model_output_path = "/opt/ml/model/model.pt"
    labels_output_path = "/opt/ml/model/labels.txt"
    class_count_path = "/opt/ml/model/class_count.txt"
    train_path = kwargs["train_path"]
    img_size = int(kwargs["img_size"])
    try:
        batch_size = int(kwargs["batch_size"])
    except:
        batch_size = "auto"
    train_augmentation = str(kwargs["train_augmentation"]).lower() == "true"
    crop = str(kwargs["crop"]).lower() == "true"
    class_count = int(kwargs["class_count"])
    epochs = int(kwargs["epochs"])
    best_model_pick = kwargs["best_model_pick"]
    force_class_id_name_length_match = str(kwargs["force_class_id_name_length_match"]).lower() == "true"
    gpu_count = int(kwargs["gpu_count"])
    #==========================

    print(train_path)
    print("crop", crop)

    os.makedirs(run_path, exist_ok=True)
    os.makedirs(run_path + os.sep + "cm", exist_ok=True)
    
    wandb.init(project="efficientnet-classifier")

    net = Net(class_count)
    if gpu_count > 1:
        gpu_ids = []
        for i in range(gpu_count):
            gpu_ids.append(i)
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)
    net = net.to(device)
    if batch_size == "auto":
        batch_size = check_train_batch_size(net, img_size, fraction=.4)

    trainloader, valloader, testloader = get_data_loaders(train_path, img_size, batch_size=batch_size, train_augmentation=train_augmentation, crop=crop)

    
    names = get_class_names(train_path)
    if force_class_id_name_length_match:
        assert len(names) == class_count, f'{len(names)} names found for class_count={class_count}'  # check
    else:
        if len(names) != class_count:
            print(f'{len(names)} names found for class_count={class_count}')  # check
            print("continuing training anyway")
    with open(labels_output_path, 'w') as fp:
        for item in names:
            # write each item on a new line
            fp.write("%s\n" % item)

    with open(class_count_path, "w")as fp:
        fp.write(str(class_count))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    best_acc = 0
    best_val_loss = 10000000

    wandb.watch(net, criterion, log="all", log_freq=100)

    for epoch in range(epochs):  # loop over the dataset multiple times
        # model_path = r"H:\From F\springhouse-svc-pytorch-classifier\src\scripts\local_docker\test_dir\model\model_1_.pt"
        # weights = torch.load(model_path)
        # model = Net(725, load_weights=True)
        # model.load_state_dict(weights)
        # model.to(device)
        # model.eval()

        running_loss = 0.0
        correct = 0
        total = 0
        y_true, y_pred = np.array([]), np.array([])

        for i, data in enumerate(trainloader):
            net.train()

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(device))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            # print statistics
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true = np.concatenate((y_true, labels.cpu().detach().numpy()))
            y_pred = np.concatenate((y_pred, predicted.cpu().detach().numpy()))
        try:
            accuracy = 100 * correct // total
        except:
            accuracy = 0
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        wandb.log({"loss": running_loss/len(trainloader), "epoch": epoch, "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1})
        print(f'Accuracy on training: {accuracy} %')
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / len(trainloader):.3f}')
        running_loss = 0.0
        
        correct = 0
        total = 0


        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            net.eval()
            for data in valloader:
                images, labels = data
                labels = labels.to(device)

                # calculate outputs by running images through the network
                outputs = net(images.to(device))

                loss = criterion(outputs, labels.to(device))



            # print statistics
                running_loss += loss.item()
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        try:
            accuracy = 100 * correct // total
        except:
            accuracy = 0

        print(f'Validation accuracy of the network: {accuracy} %')
        print("val loss", str(running_loss / len(valloader)))

        if best_model_pick == "val_acc" and accuracy > best_acc:
            print("val acc model saved")
            best_acc = accuracy
            if isinstance(net, torch.nn.DataParallel):
                torch.save(net.module.state_dict(), model_output_path)
            else:
                torch.save(net.state_dict(), model_output_path)
        elif best_model_pick == "val_loss" and running_loss < best_val_loss:
            print("val loss model saved")
            best_val_loss = running_loss
            if isinstance(net, torch.nn.DataParallel):
                torch.save(net.module.state_dict(), model_output_path)
            else:
                torch.save(net.state_dict(), model_output_path)
    
    # upload model
    artifact = wandb.Artifact('best-model', type='model')
    artifact.add_file(model_output_path)
    wandb.log_artifact(artifact)

    try:
        artifact = wandb.Artifact('model_config_class_count')
        artifact.add_file(class_count_path)
        wandb.log_artifact(artifact)


        artifact = wandb.Artifact('model_config_class_names')
        artifact.add_file(labels_output_path)
        wandb.log_artifact(artifact)
    except Exception as e:
       print(e)
    print('Finished Training')

if __name__ == "__main__":
    import json
    import time
    start_time = time.time()



    prefix = os.path.join("src", "scripts", "local_docker", "test_dir")
    param_path = os.path.join(prefix, "input", "config", "hyperparameters.json")

        # Read in any hyperparameters that the user passed with the training job
    with open(param_path, 'r') as tc:
        trainingParams = json.load(tc)
    train_main(**trainingParams)
    print("--- %s seconds ---" % (time.time() - start_time))
