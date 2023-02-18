import os

dir_path = 'DummyDatabase'
marmot_v1 = dir_path + '/marmot_v1'
marmot_extended = dir_path + '/marmot_extended'
ORIG_DATA_PATH = f'{marmot_v1}/marmot_dataset_v1.0/data/English'
DATA_PATH = 'Marmot_data'
PROCESSED_DATA = f'{dir_path}/marmot_processed'
PREDICTIONS = f"{dir_path}/predictions"
TEST_IMAGES = f"{dir_path}/test_images"
MODELS = f"{dir_path}/models"
IMAGE_PATH = os.path.join(PROCESSED_DATA, 'image')
TABLE_MASK_PATH = os.path.join(PROCESSED_DATA, 'table_mask')
COL_MASK_PATH = os.path.join(PROCESSED_DATA, 'col_mask')
Marmot_data = f'{dir_path}/{DATA_PATH}'
POSITIVE_DATA_LBL = os.path.join(ORIG_DATA_PATH, 'Positive','Labeled')
