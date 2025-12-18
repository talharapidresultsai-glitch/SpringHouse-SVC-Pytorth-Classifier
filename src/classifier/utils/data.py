import os
import random
from re import X
from PIL import Image
import torch
import torchvision
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import albumentations as A

class bb_zoom_dataset(Dataset):
    def __init__(self, img_root_dir, labels_root_dir, img_size=256, data_transform=None, augmentation=False, crop=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = img_root_dir
        self.sample_paths = os.listdir(img_root_dir)
        
        self.labels_root_dir = labels_root_dir
        self.data_transform = data_transform

        self.img_size = img_size
        self.augment = augmentation
        self.crop = crop

        if augmentation:
            prob=.1
            self.transform = A.Compose([
                A.GridDropout(p=prob),
                A.OpticalDistortion (p=prob),
                A.PixelDropout (p=prob),
                #A.RandomCrop (p=prob),
                A.RandomScale (p=prob),
                A.Blur(p=prob),
                #A.CLAHE (p=prob),
                A.ColorJitter (p=prob),
                A.FancyPCA(p=prob),
                A.GaussianBlur (p=prob),
                # A.GaussNoise (p=prob),
                # A.GlassBlur (p=prob),
                # A.HueSaturationValue (p=prob),
                # A.ImageCompression (p=prob),
                # A.MedianBlur(p=prob),
                # A.OpticalDistortion (p=prob),
                A.RandomFog (p=prob),
                A.RandomGamma(p=prob),
                A.RandomShadow (p=prob),
                A.Sharpen (p=prob),
                A.Superpixels (p=prob),
                A.RandomBrightnessContrast(p=prob),
            ])
            

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample_path = self.sample_paths[idx]
        path = os.path.join(self.root_dir, sample_path)
        img = Image.open(path)

        if self.augment:
            img = Image.fromarray(self.transform(image=np.array(img))["image"])
            

        class_id, x_min, y_min, x_max, y_max = self.get_label(sample_path)
        if self.crop:
            width, height = img.size

            x_min = x_min * width
            y_min = y_min * height

            x_max = x_max * width
            y_max = y_max * height

            if self.augment:
            

                percent_crop = .25
                
                min_x_change = (random.uniform(0, 1) * 2 * percent_crop) - percent_crop
                min_y_change = (random.uniform(0, 1) * 2 * percent_crop) - percent_crop

                max_x_change = (random.uniform(0, 1) * 2 * percent_crop) - percent_crop
                max_y_change = (random.uniform(0, 1) * 2 * percent_crop) - percent_crop

                crop_width = x_max - x_min
                crop_height = y_max - y_min

                new_x_min = x_min - (crop_width * min_x_change)
                new_y_min = y_min - (crop_height * min_y_change)

                new_x_max = x_max - (crop_width *  max_x_change)
                new_y_max = y_max - (crop_height *  max_y_change)

                new_x_min = int(max(new_x_min, 0))
                new_y_min = int(max(new_y_min, 0))

                new_x_max = int(min(new_x_max, width))
                new_y_max = int(min(new_y_max, height))

                if new_x_min != new_x_max and new_y_min != new_y_max:
                    if new_x_min < new_x_max:
                        x_min = new_x_min
                        x_max = new_x_max
                    else:
                        x_max = new_x_min
                        x_min = new_x_max
                    
                    if new_y_min < new_y_max:
                        y_min = new_y_min
                        y_max = new_y_max
                    else:
                        y_max = new_x_min
                        y_min = new_y_max

            img = img.crop((x_min, y_min, x_max, y_max))
        img = resize(img, self.img_size, self.img_size)

        #img.save("test.jpg")


        if self.data_transform:
            img = self.data_transform(img)
        
        return img, class_id

    def get_label(self, sample_path):
                #get label
        label_path = os.path.join(self.labels_root_dir, sample_path).replace(".jpg", ".txt")
        print(label_path)
        with open(label_path, "r") as file:
            label = file.readline()
            label = label.strip()
        
        #convert data types
        class_id, x, y, wi, h = label.split(" ")
        class_id = int(class_id)

        x, y, wi, h = float(x), float(y), float(wi), float(h)

        half_wi = wi/2
        half_h = h/2

        x_min = x - half_wi
        x_max = x + half_wi

        y_min = y - half_h
        y_max = y + half_h

        return class_id, x_min, y_min, x_max, y_max
    
def resize(image_pil, width, height):
    '''
    Resize PIL image keeping ratio and using white background.
    '''
    ratio_w = width / image_pil.width
    ratio_h = height / image_pil.height
    if ratio_w < ratio_h:
        # It must be fixed by width
        resize_width = width
        resize_height = round(ratio_w * image_pil.height)
    else:
        # Fixed by height
        resize_width = round(ratio_h * image_pil.width)
        resize_height = height
    image_resize = image_pil.resize((resize_width, resize_height), Image.ANTIALIAS)
    background = Image.new('RGBA', (width, height), (127, 127, 127, 255))
    offset = (round((width - resize_width) / 2), round((height - resize_height) / 2))
    background.paste(image_resize, offset)
    return background.convert('RGB')



        


def get_data_loaders(path, img_size, this_transform = None, batch_size=32, train_augmentation=False, crop=True):

    this_transform = transforms.Compose(
        [transforms.ToTensor()])

    train_path = os.path.join(path, "images", "train")
    train_labels_path = os.path.join(path, "labels", "train")
    trainset = bb_zoom_dataset(train_path, train_labels_path, img_size, data_transform=this_transform, augmentation=train_augmentation, crop=crop)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True)

    val_path = os.path.join(path, "images", "val")
    val_lables_path = os.path.join(path, "labels", "val")
    valset = bb_zoom_dataset(val_path, val_lables_path, img_size, data_transform=this_transform, crop=crop)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=False)

    test_path = os.path.join(path, "images", "test")
    test_labels_path = os.path.join(path, "labels", "test")
    testset = bb_zoom_dataset(test_path, test_labels_path, img_size, data_transform=this_transform, crop=crop)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False)

 
    return trainloader, valloader, testloader

def save_model_input_to_disk(input, name="train_"):
    import uuid
    single_image = input[0, ...].cpu().numpy()
    unregged = single_image * 255
    tped = np.transpose(unregged, [1, 2, 0])
    img = Image.fromarray(tped.astype("uint8"))
    
    id = uuid.uuid4()
    out_folder = "save_to_disk"
    os.makedirs(out_folder, exist_ok=True)
    img.save(out_folder + os.sep + name +str(id) + ".jpg")

