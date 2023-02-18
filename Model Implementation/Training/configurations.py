from Training.path_constants import PROCESSED_DATA
import torch # pip install torch

SEED = 0
LEARNING_RATE = 0.0001
EPOCHS = 100
BATCH_SIZE = 2
WEIGHT_DECAY = 3e-4
DATAPATH = f'{PROCESSED_DATA}/processed_data.csv'
MODEL_NAME = "densenet_configuration_4_model_checkpoint.pth.tar"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
