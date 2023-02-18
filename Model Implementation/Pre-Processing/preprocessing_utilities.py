import struct
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET


def get_table_bounding_box(table_xml_path: str,  new_image_shape: tuple):
    """
        Goal: Extract table coordinates from xml file and scale them to the new image shape
        Input:
            :param table_xml_path: xml file path
            :param new_image_shape: tuple (new_h, new_w)
        Return: table_bounding_boxes: List of all the bounding boxes of the tables
    """
    tree = ET.parse(table_xml_path)
    root = tree.getroot()
    left, top, right, bottom = list(map(lambda x: struct.unpack('!d', bytes.fromhex(x))[0], root.get("CropBox").split()))
    width = abs(right - left)
    height = abs(top - bottom)
    table_bounding_boxes = []
    for table in root.findall(".//Composite[@Label='TableBody']"):
        x0in, y0in, x1in, y1in  = list(map(lambda x: struct.unpack('!d', bytes.fromhex(x))[0], table.get("BBox").split()))
        x0 = round(new_image_shape[1] * (x0in - left) / width)
        x1 = round(new_image_shape[1] * (x1in - left) / width)
        y0 = round(new_image_shape[0] * (top - y0in) / height)
        y1 = round(new_image_shape[0] * (top - y1in) / height)
        table_bounding_boxes.append([x0, y0, x1, y1])
    return table_bounding_boxes

def get_column_bounding_box(column_xml_path: str, old_image_shape: tuple, new_image_shape: tuple, 
                                                    table_bounding_box: list, threshhold: int = 3):
    """
        Goal: 
            - Extract column coordinates from the xml file and scale them to the new image shape and the old image shape
            - If there are no table_bounding_box present, approximate them using column bounding box
        Input:
            :param table_xml_path: xml file path
            :param old_image_shape: (new_h, new_w)
            :param new_image_shape: (new_h, new_w)
            :param table_bounding_box: List of table bbox coordinates
            :param threshold: the threshold t apply, defualts to 3
        Return: tuple (column_bounding_box, table_bounding_box)
    """
    tree = ET.parse(column_xml_path)
    root = tree.getroot()
    x_mins = [round(int(coord.text) * new_image_shape[1] / old_image_shape[1]) for coord in root.findall("./object/bndbox/xmin")]
    y_mins = [round(int(coord.text) * new_image_shape[0] / old_image_shape[0]) for coord in root.findall("./object/bndbox/ymin")]
    x_maxs = [round(int(coord.text) * new_image_shape[1] / old_image_shape[1]) for coord in root.findall("./object/bndbox/xmax")]
    y_maxs = [round(int(coord.text) * new_image_shape[0] / old_image_shape[0]) for coord in root.findall("./object/bndbox/ymax")]
    column_bounding_box = []
    for x_min, y_min, x_max, y_max in zip(x_mins, y_mins, x_maxs, y_maxs):
        bounding_box = [x_min, y_min, x_max, y_max]
        column_bounding_box.append(bounding_box)
    if len(table_bounding_box) == 0:
        x_min = min([x[0] for x in column_bounding_box]) - threshhold
        y_min = min([x[1] for x in column_bounding_box]) - threshhold
        x_max = max([x[2] for x in column_bounding_box]) + threshhold
        y_max = max([x[3] for x in column_bounding_box]) + threshhold
        table_bounding_box = [[x_min, y_min, x_max, y_max]]
    return column_bounding_box, table_bounding_box

def create_element_mask(new_h: int, new_w: int, bounding_boxes: list = None):
    """
        Goal: Create a mask based on new_h, new_w and bounding boxes
        Input:
            :param new_h: height of the mask
            :param new_w: width of the mask
            :param bounding_boxes:  bounding box coordinates  
        Return: mask: Image 
    """
    mask = np.zeros((new_h, new_w), dtype = np.int32)
    if bounding_boxes is None or len(bounding_boxes) == 0:
         return Image.fromarray(mask)
    for box in bounding_boxes:
        mask[box[1]:box[3], box[0]:box[2]] = 255
    return Image.fromarray(mask)
