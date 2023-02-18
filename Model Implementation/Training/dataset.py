import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import albumentations as A 
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import tqdm
from Training.path_constants import PROCESSED_DATA


class ImageFolder(nn.Module):
    def __init__(self, df, transform = None):
        super(ImageFolder, self).__init__()
        self.df = df
        if transform is None:
            self.transform = A.Compose([
                A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 255,),
                ToTensorV2()
            ])
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path, table_mask_path, column_mask_path = self.df.iloc[index, 0], self.df.iloc[index, 1], self.df.iloc[index, 2]
        image = np.array(Image.open(image_path))
        table_image = torch.FloatTensor(np.array(Image.open(table_mask_path)) / 255.0).reshape(1, 1024, 1024)
        column_image = torch.FloatTensor(np.array(Image.open(column_mask_path)) / 255.0).reshape(1, 1024, 1024)
        image = self.transform(image = image)['image']
        return {"image": image, "table_image": table_image, "column_image": column_image}

def get_mean_std(train_data, transform):
    dataset = ImageFolder(train_data , transform)
    train_loader = DataLoader(dataset, batch_size = 128)
    mean = 0.
    std = 0.
    for img_dict in tqdm.tqdm(train_loader):
        batch_samples = img_dict["image"].size(0)
        images = img_dict["image"].view(batch_samples, img_dict["image"].size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= len(train_loader.dataset)
    std /= len(train_loader.dataset)
    print(mean)
    print(std)

# Read referencing csv file
df = pd.read_csv(f'{PROCESSED_DATA}/processed_data.csv')
dataset = ImageFolder(df[df['hasTable'] == 1])
img_num = 0
for img_dict in dataset:
    save_image(img_dict["image"], f'image_{img_num}.png')
    save_image(img_dict["table_image"], f'table_image_{img_num}.png')
    save_image(img_dict["column_image"], f'column_image_{img_num}.png')
    img_num += 1
    if img_num == 6:
        break
