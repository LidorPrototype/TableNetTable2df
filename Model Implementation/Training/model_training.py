import time
import torch
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary

from Training.configurations import DATAPATH, DEVICE, EPOCHS, LEARNING_RATE, MODEL_NAME, SEED, WEIGHT_DECAY, BATCH_SIZE
from Training.path_constants import PROCESSED_DATA
from Training.tablenet_model import TableNet
from Training.model_loss import TableNetLoss
from Training.general_utilities import compute_metrics, seed_all, get_data_loaders, load_checkpoint, display_metrics, write_summary, save_checkpoint

import warnings
warnings.filterwarnings("ignore")


def train_on_epoch(data_loader, model, optimizer, loss, scaler, threshold = 0.5):
    combined_loss = []
    table_loss, table_acc, table_precision, table_recall, table_f1 = [], [], [], [], []
    column_loss, column_acc, column_precision, column_recall, column_f1 = [], [], [], [], []
    loop = tqdm(data_loader, leave = True)
    for batch_i, image_dict in enumerate(loop):
        image            = image_dict["image"].to(DEVICE)
        table_image      = image_dict["table_image"].to(DEVICE)
        column_image     = image_dict["column_image"].to(DEVICE)
        with torch.cuda.amp.autocast():
            table_out, column_out = model(image)
            i_table_loss, i_column_loss = loss(table_out, table_image, column_out, column_image)
        table_loss.append(i_table_loss.item())
        column_loss.append(i_column_loss.item())
        combined_loss.append((i_table_loss + i_column_loss).item())
        # Backward
        optimizer.zero_grad()
        scaler.scale(i_table_loss + i_column_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        mean_loss = sum(combined_loss) / len(combined_loss)
        loop.set_postfix(loss = mean_loss)
        cal_metrics_table = compute_metrics(table_image, table_out, threshold)
        cal_metrics_col = compute_metrics(column_image, column_out, threshold)
        table_f1.append(cal_metrics_table['f1'])
        table_precision.append(cal_metrics_table['precision'])
        table_acc.append(cal_metrics_table['acc'])
        table_recall.append(cal_metrics_table['recall'])
        column_f1.append(cal_metrics_col['f1'])
        column_acc.append(cal_metrics_col['acc'])
        column_precision.append(cal_metrics_col['precision'])
        column_recall.append(cal_metrics_col['recall'])
        metrics = {
          'combined_loss': np.mean(combined_loss),
          'table_loss': np.mean(table_loss),
          'column_loss': np.mean(column_loss),
          'table_acc': np.mean(table_acc),
          'col_acc': np.mean(column_acc),
          'table_f1': np.mean(table_f1),
          'col_f1': np.mean(column_f1),
          'table_precision': np.mean(table_precision),
          'col_precision': np.mean(column_precision),
          'table_recall': np.mean(table_recall),
          'col_recall': np.mean(column_recall)
        }
    return metrics

def test_on_epoch(data_loader, model, loss, threshold = 0.5, device = DEVICE):
    combined_loss = []
    table_loss, table_acc, table_precision, table_recall, table_f1 = [], [], [], [], []
    column_loss, column_acc, column_precision, column_recall, column_f1 = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        loop = tqdm(data_loader, leave = True)
        for batch_i, image_dict in enumerate(loop):
            image            = image_dict["image"].to(device)
            table_image      = image_dict["table_image"].to(device)
            column_image     = image_dict["column_image"].to(device)
            with torch.cuda.amp.autocast():
                table_out, column_out  = model(image)
                i_table_loss, i_column_loss = loss(table_out, table_image, column_out, column_image)
            table_loss.append(i_table_loss.item())
            column_loss.append(i_column_loss.item())
            combined_loss.append((i_table_loss + i_column_loss).item())
            mean_loss = sum(combined_loss) / len(combined_loss)
            loop.set_postfix(loss=mean_loss)
            cal_metrics_table = compute_metrics(table_image, table_out, threshold)
            cal_metrics_col = compute_metrics(column_image, column_out, threshold)
            table_f1.append(cal_metrics_table['f1'])
            table_precision.append(cal_metrics_table['precision'])
            table_acc.append(cal_metrics_table['acc'])
            table_recall.append(cal_metrics_table['recall'])
            column_f1.append(cal_metrics_col['f1'])
            column_acc.append(cal_metrics_col['acc'])
            column_precision.append(cal_metrics_col['precision'])
            column_recall.append(cal_metrics_col['recall'])
    metrics = {
        'combined_loss': np.mean(combined_loss),
        'table_loss': np.mean(table_loss),
        'column_loss': np.mean(column_loss),
        'table_acc': np.mean(table_acc),
        'col_acc': np.mean(column_acc),
        'table_f1': np.mean(table_f1),
        'col_f1': np.mean(column_f1),
        'table_precision': np.mean(table_precision),
        'col_precision': np.mean(column_precision),
        'table_recall': np.mean(table_recall),
        'col_recall': np.mean(column_recall)
    }
    model.train()
    return metrics

seed_all(SEED_VALUE = SEED)
checkpoint_name = f'{PROCESSED_DATA}/{MODEL_NAME}'
model = TableNet(encoder = 'densenet', use_pretrained_model = True, basemodel_requires_grad = True)

print("Model Architecture and Trainable Paramerters")
print("="*50)
print(summary(model, torch.zeros((1, 3, 1024, 1024)), show_input = False, show_hierarchical = True))

model = model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay = WEIGHT_DECAY)
loss = TableNetLoss()
scaler = torch.cuda.amp.GradScaler()
train_loader, test_loader = get_data_loaders(data_path = DATAPATH)

# Load checkpoint
if os.path.exists(checkpoint_name):
    last_epoch, train_metrics, test_metrics = load_checkpoint(torch.load(checkpoint_name), model)
    last_table_f1 = test_metrics['table_f1']
    last_column_f1 = test_metrics['col_f1']
    print("Loading Checkpoint...")
    display_metrics(last_epoch, train_metrics, test_metrics)
    print()
else:
    last_epoch = 0
    last_table_f1 = 0.
    last_column_f1 = 0.

# Train Network
print("Training Model\n")
writer = SummaryWriter(f"{PROCESSED_DATA}/runs/TableNet/densenet/configuration_4_batch_{BATCH_SIZE}_learningrate_{LEARNING_RATE}_encoder_train")
# For early stopping
i = 0

for epoch in range(last_epoch + 1, EPOCHS):
    print("="*30)
    start = time.time()
    train_metrics = train_on_epoch(train_loader, model, optimizer, loss, scaler, threshold = 0.5)
    test_metrics = test_on_epoch(test_loader, model, loss, threshold = 0.5)
    write_summary(writer, train_metrics, test_metrics, epoch)
    end = time.time()
    display_metrics(epoch, train_metrics, test_metrics)
    if last_table_f1 < test_metrics['table_f1'] or last_column_f1 < test_metrics['col_f1']:
        last_table_f1 = test_metrics['table_f1']
        last_column_f1 = test_metrics['col_f1']
        checkpoint = {
            'epoch': epoch, 
            'state_dict': model.state_dict(), 
            'optimizer': optimizer.state_dict(), 
            'train_metrics': train_metrics, 
            'test_metrics': test_metrics
        }
        save_checkpoint(checkpoint, checkpoint_name)
