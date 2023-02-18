import os
import glob
from tqdm import tqdm
from PIL import Image
import pandas as pd
from Training.path_constants import ORIG_DATA_PATH, PROCESSED_DATA, IMAGE_PATH, TABLE_MASK_PATH, COL_MASK_PATH, POSITIVE_DATA_LBL, DATA_PATH
from preprocessing_utilities import get_table_bounding_box, get_column_bounding_box, create_element_mask


# Make directories to save data
os.makedirs(PROCESSED_DATA, exist_ok = True)
os.makedirs(IMAGE_PATH, exist_ok = True)
os.makedirs(TABLE_MASK_PATH, exist_ok = True)
os.makedirs(COL_MASK_PATH, exist_ok = True)

positive_data = glob.glob(f'{ORIG_DATA_PATH}/Positive/Raw' + '/*.bmp')
negative_data = glob.glob(f'{ORIG_DATA_PATH}/Negative/Raw' + '/*.bmp')

new_h, new_w = 1024, 1024

processed_data = []
for i, data in enumerate([negative_data, positive_data]):
    for j, image_path in tqdm(enumerate(data)):
        image_name = os.path.basename(image_path)
        image = Image.open(image_path)
        w, h = image.size
        # Convert image to RGB image
        image = image.resize((new_h, new_w))
        if image.mode != 'RGB':
            image = image.convert("RGB")
        table_bounding_boxes, column_bounding_boxes = [], []
        if i == 1:
            # Get xml filename
            xml_file = image_name.replace('bmp', 'xml')
            table_xml_path = os.path.join(POSITIVE_DATA_LBL, xml_file)
            column_xml_path = os.path.join(DATA_PATH, xml_file)
            # Get bounding boxes
            table_bounding_boxes = get_table_bounding_box(table_xml_path, (new_h, new_w))
            if os.path.exists(column_xml_path):
                column_bounding_boxes, table_bounding_boxes = get_column_bounding_box(column_xml_path, (h,w), (new_h, new_w), table_bounding_boxes)
            else:
                column_bounding_boxes = []
        # Create masks
        table_mask = create_element_mask(new_h, new_w, table_bounding_boxes)
        column_mask = create_element_mask(new_h, new_w, column_bounding_boxes)
        # Save images and masks
        save_image_path = os.path.join(IMAGE_PATH, image_name.replace('bmp', 'jpg'))
        save_table_mask_path = os.path.join(TABLE_MASK_PATH, image_name[:-4] + '_table_mask.png')
        save_column_mask_path = os.path.join(COL_MASK_PATH, image_name[:-4] + '_col_mask.png')
        image.save(save_image_path)
        table_mask.save(save_table_mask_path)
        column_mask.save(save_column_mask_path)
        # Add data to the dataframe
        len_table = len(table_bounding_boxes)
        len_columns = len(column_bounding_boxes)
        value = (save_image_path, save_table_mask_path, save_column_mask_path, h, w, int(len_table != 0), \
                 len_table, len_columns, table_bounding_boxes, column_bounding_boxes)
        processed_data.append(value)

columns_name = ['img_path', 'table_mask', 'col_mask', 'original_height', 'original_width', 'hasTable', 'table_count', 'col_count', 'table_bboxes', 'col_bboxes']
processed_data = pd.DataFrame(processed_data, columns=columns_name)
# Save dataframe and inspect it's data
processed_data.to_csv(f"{PROCESSED_DATA}/processed_data.csv", index = False)
print(processed_data.tail())
