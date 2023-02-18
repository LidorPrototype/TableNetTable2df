import random
import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A 
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from albumentations.pytorch import ToTensorV2

from Training.configurations import BATCH_SIZE, DATAPATH, DEVICE, SEED
from Training.path_constants import PROCESSED_DATA
from Training.dataset import ImageFolder


TRANSFORM = A.Compose([
    A.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], max_pixel_value = 255,),
    ToTensorV2()
])
# Apply the SEED
def seed_all(SEED_VALUE = SEED):
    random.seed(SEED_VALUE)
    os.environ['PYTHONHASHSEED'] = str(SEED_VALUE)
    np.random.seed(SEED_VALUE)
    torch.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed(SEED_VALUE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_data_loaders(data_path = DATAPATH):
    df = pd.read_csv(data_path)
    train_data, test_data  = train_test_split(df, test_size = 0.2, random_state = SEED, stratify = df.hasTable)
    train_dataset = ImageFolder(train_data, isTrain = True, transform = None)
    test_dataset = ImageFolder(test_data, isTrain = False, transform = None)
    train_loader =  DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4, pin_memory = True)
    test_loader =  DataLoader(test_dataset, batch_size = 8, shuffle = False, num_workers = 4, pin_memory = True)
    return train_loader, test_loader

# Save Checkpoint
def save_checkpoint(state, filename = f"{PROCESSED_DATA}/model_checkpoint.pth.tar"):
    torch.save(state, filename)
    print("Checkpoint Saved at: ", filename)

# Load the checkpoint we saved
def load_checkpoint(checkpoint, model, optimizer = None):
    print("Loading checkpoint...")
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    last_epoch = checkpoint['epoch']
    tr_metrics = checkpoint['train_metrics']
    te_metrics = checkpoint['test_metrics']
    return last_epoch, tr_metrics, te_metrics

def write_summary(writer, tr_metrics, te_metrics, epoch):
    writer.add_scalar("Table Loss/Train", tr_metrics['table_loss'], global_step = epoch)
    writer.add_scalar("Table Loss/Test", te_metrics['table_loss'], global_step = epoch)
    writer.add_scalar("Table Acc/Train", tr_metrics['table_acc'], global_step = epoch)
    writer.add_scalar("Table Acc/Test", te_metrics['table_acc'], global_step = epoch)
    writer.add_scalar("Table F1/Train", tr_metrics['table_f1'], global_step = epoch)
    writer.add_scalar("Table F1/Test", te_metrics['table_f1'], global_step = epoch)
    writer.add_scalar("Table Precision/Train", tr_metrics['table_precision'], global_step = epoch)
    writer.add_scalar("Table Precision/Test", te_metrics['table_precision'], global_step = epoch)
    writer.add_scalar("Table Recall/Train", tr_metrics['table_recall'], global_step = epoch)
    writer.add_scalar("Table Recall/Test", te_metrics['table_recall'], global_step = epoch)
    writer.add_scalar("Column Loss/Train", tr_metrics['column_loss'], global_step = epoch)
    writer.add_scalar("Column Loss/Test", te_metrics['column_loss'], global_step = epoch)
    writer.add_scalar("Column Acc/Train", tr_metrics['col_acc'], global_step = epoch)
    writer.add_scalar("Column Acc/Test", te_metrics['col_acc'], global_step = epoch)
    writer.add_scalar("Column F1/Train", tr_metrics['col_f1'], global_step = epoch)
    writer.add_scalar("Column F1/Test", te_metrics['col_f1'], global_step = epoch)    
    writer.add_scalar("Column Precision/Train", tr_metrics['col_precision'], global_step = epoch)
    writer.add_scalar("Column Precision/Test", te_metrics['col_precision'], global_step = epoch)
    writer.add_scalar("Column Recall/Train", tr_metrics['col_recall'], global_step = epoch)
    writer.add_scalar("Column Recall/Test", te_metrics['col_recall'], global_step = epoch)

def display_metrics(epoch, tr_metrics, te_metrics):
    print(f"Epoch: {epoch} \n\
        Table Loss -- Train: {tr_metrics['table_loss']:.3f} Test: {te_metrics['table_loss']:.3f}\n\
        Table Acc -- Train: {tr_metrics['table_acc']:.3f} Test: {te_metrics['table_acc']:.3f}\n\
        Table F1 -- Train: {tr_metrics['table_f1']:.3f} Test: {te_metrics['table_f1']:.3f}\n\
        Table Precision -- Train: {tr_metrics['table_precision']:.3f} Test: {te_metrics['table_precision']:.3f}\n\
        Table Recall -- Train: {tr_metrics['table_recall']:.3f} Test: {te_metrics['table_recall']:.3f}\n\
        \n\
        Col Loss -- Train: {tr_metrics['column_loss']:.3f} Test: {te_metrics['column_loss']:.3f}\n\
        Col Acc -- Train: {tr_metrics['col_acc']:.3f} Test: {te_metrics['col_acc']:.3f}\n\
        Col F1 -- Train: {tr_metrics['col_f1']:.3f} Test: {te_metrics['col_f1']:.3f}\n\
        Col Precision -- Train: {tr_metrics['col_precision']:.3f} Test: {te_metrics['col_precision']:.3f}\n\
        Col Recall -- Train: {tr_metrics['col_recall']:.3f} Test: {te_metrics['col_recall']:.3f}\n"
    )

def compute_metrics(ground_truth, prediction, threshold = 0.5):
    # Ref: https://stackoverflow.com/a/56649983
    ground_truth = ground_truth.int()
    prediction = (torch.sigmoid(prediction) > threshold).int()
    TP = torch.sum(prediction[ground_truth == 1] == 1)
    TN = torch.sum(prediction[ground_truth == 0] == 0)
    FP = torch.sum(prediction[ground_truth == 1] == 0)
    FN = torch.sum(prediction[ground_truth == 0] == 1)
    acc = (TP + TN) / (TP + TN + FP+ FN)
    precision = TP / (FP + TP + 1e-4)
    recall = TP / (FN + TP + 1e-4)
    f1 = 2 * precision * recall / (precision + recall + 1e-4)
    metrics = {
        'acc': acc.item(),
        'f1': f1.item(),
        'precision':precision.item(),
        'recall': recall.item()
    }
    return metrics

def display(image, table, column, title = 'Original'):
    f, ax  = plt.subplots(1, 3, figsize = (15, 8))
    ax[0].imshow(image)
    ax[0].set_title(f'{title} Image')
    ax[1].imshow(table)
    ax[1].set_title(f'{title} Table Mask')
    ax[2].imshow(column)
    ax[2].set_title(f'{title} Column Mask')
    plt.show()

def display_prediction(image, table = None, table_image = None, no_: bool = False):
  if no_:
    f1, ax  = plt.subplots(1, 1, figsize = (7, 5))
    ax.imshow(image)
    ax.set_title('Original Image')
    f1.suptitle('No Tables Detected')
  else:
    f2, ax  = plt.subplots(1, 3, figsize = (15, 8))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[1].imshow(table)
    ax[1].set_title('Image with Predicted Table')
    ax[2].imshow(table_image)
    ax[2].set_title('Predicted Table Example')
  plt.show()

def get_TableMasks(test_image, model, transform = TRANSFORM, device = DEVICE):
    image = transform(image = test_image)["image"]
    # Get predictions
    model.eval()
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)
        # With torch.cuda.amp.autocast():
        table_out, column_out  = model(image)
        table_out = torch.sigmoid(table_out)
        column_out = torch.sigmoid(column_out)
    # Remove gradients
    table_out = (table_out.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0) > 0.5).astype(int)
    column_out = (column_out.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0) > 0.5).astype(int)
    # Return masks
    return table_out, column_out

def fixMasks(image, table_mask, column_mask):
    """ Fix Table Bounding Box to get better OCR predictions """
    table_mask = table_mask.reshape(1024, 1024).astype(np.uint8)
    column_mask = column_mask.reshape(1024, 1024).astype(np.uint8)
    # Get contours of the mask to get number of tables
    contours, table_heirarchy = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    table_contours = []
    # Ref: https://www.pyimagesearch.com/2015/02/09/removing-contours-image-using-python-opencv/
    # Remove bad contours
    for c in contours:
        # if the contour is bad, draw it on the mask
        if cv2.contourArea(c) > 2000:
            table_contours.append(c)
    if len(table_contours) == 0:
        return None
    # Ref : https://docs.opencv.org/4.5.2/da/d0c/tutorial_bounding_rects_circles.html
    # Get bounding box for the contour
    table_bound_rect = [None] * len(table_contours)
    for i, c in enumerate(table_contours):
        polygon = cv2.approxPolyDP(c, 3, True)
        table_bound_rect[i] = cv2.boundingRect(polygon)
    # Table bounding Box
    table_bound_rect.sort()
    column_bound_rects = []
    for x, y, w, h in table_bound_rect:
        column_mask_crop = column_mask[y : y + h, x : x + w]
        # Get contours of the mask to get number of tables
        contours, column_heirarchy = cv2.findContours(column_mask_crop, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Get bounding box for the contour
        bound_rect = [None] * len(contours)
        for i, c in enumerate(contours):
            polygon = cv2.approxPolyDP(c, 3, True)
            bound_rect[i] = cv2.boundingRect(polygon)
            # Adjusting columns as per table coordinates
            bound_rect[i] = (bound_rect[i][0] + x, bound_rect[i][1] + y, bound_rect[i][2], bound_rect[i][3])
        column_bound_rects.append(bound_rect)
    image = image[...,0].reshape(1024, 1024).astype(np.uint8)
    # Draw bounding boxes
    color = (0, 255, 0)
    thickness = 4
    for x, y, w, h in table_bound_rect:
        image = cv2.rectangle(image, (x, y),(x + w, y + h), color, thickness)
    return image, table_bound_rect, column_bound_rects
