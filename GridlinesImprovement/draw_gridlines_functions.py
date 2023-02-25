import cv2
import numpy as np


"""
  Return:
    (dict):
        'threshold': threshold image
        'vertical': vertical grid lines image
        'horizontal': horizontal grid lines image
        'full': full grid lines image
"""
def drawGridlines(old_image: np.ndarray) -> dict:
  # Get dimensions
  hh_, ww_ = old_image.shape[:2]
  # Convert image to grayscale 
  gray = cv2.cvtColor(old_image, cv2.COLOR_BGR2GRAY)
  # Threshold on white - binary
  thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]
  # Resize thresh image to a single row
  row = cv2.resize(thresh, (ww_, 1), interpolation = cv2.INTER_AREA)
  # Threshold on white
  thresh_row = cv2.threshold(row, 254, 255, cv2.THRESH_BINARY)[1]
  # Apply small amount of morphology to merge with column of text
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (5, 1))
  thresh_row = cv2.morphologyEx(thresh_row, cv2.MORPH_OPEN, kernel)
  # Get vertical contours
  contours_v = cv2.findContours(thresh_row, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours_v = contours_v[0] if len(contours_v) == 2 else contours_v[1]
  full_grid_image = old_image.copy()
  vertical_img = old_image.copy()
  for contour in contours_v:
      x, y, w, h = cv2.boundingRect(contour)
      xcenter = x + w // 2
      cv2.line(vertical_img, (xcenter, 0), (xcenter, hh_ - 1), (0, 0, 0), 1)
      cv2.line(full_grid_image, (xcenter, 0), (xcenter, hh_ - 1), (0, 0, 0), 1)
  # Resize thresh image to a single column
  column = cv2.resize(thresh, (1, hh_), interpolation = cv2.INTER_AREA)
  # Threshold on white - binary
  thresh_column = cv2.threshold(column, 254, 255, cv2.THRESH_BINARY)[1]
  # Get horizontal contours
  contours_h = cv2.findContours(thresh_column, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours_h = contours_h[0] if len(contours_h) == 2 else contours_h[1]
  horizontal_img = old_image.copy()
  for contour in contours_h:
      x, y, w, h = cv2.boundingRect(contour)
      ycenter = y + h // 2
      cv2.line(horizontal_img, (0, ycenter), (ww_ - 1, ycenter), (0, 0, 0), 1)
      cv2.line(full_grid_image, (0, ycenter), (ww_ - 1, ycenter), (0, 0, 0), 1)
  # Return results as a dictionary
  return {
    'threshold': thresh,
    'vertical': vertical_img,
    'horizontal': horizontal_img,
    'full': full_grid_image
  }
