import cv2
import numpy as np

def drawGridlines(img: np.ndarray) -> dict:
  # Get dimensions
  hh, ww = img.shape[:2]
  # convert to grayscale 
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # threshold on white
  thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)[1]
  # average gray image to one row
  row = cv2.resize(thresh, (ww, 1), interpolation = cv2.INTER_AREA)
  # threshold on white
  thresh1 = cv2.threshold(row, 254, 255, cv2.THRESH_BINARY)[1]
  # apply small amount of morphology to merge period with column of text
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (5,1))
  thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
  # get contours
  contours = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1]
  full_grid_image = img.copy()
  vertical_img = img.copy()
  for cntr in contours:
      x, y, w, h = cv2.boundingRect(cntr)
      xcenter = x + w // 2
      cv2.line(vertical_img, (xcenter,0), (xcenter, hh-1), (0, 0, 0), 1)
      cv2.line(full_grid_image, (xcenter,0), (xcenter, hh-1), (0, 0, 0), 1)
  # average gray image to one column
  column = cv2.resize(thresh, (1, hh), interpolation = cv2.INTER_AREA)
  # threshold on white
  thresh2 = cv2.threshold(column, 254, 255, cv2.THRESH_BINARY)[1]
  # get contours
  contours = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1]
  horizontal_img = img.copy()
  for cntr in contours:
      x, y, w, h = cv2.boundingRect(cntr)
      ycenter = y + h // 2
      cv2.line(horizontal_img, (0, ycenter), (ww-1, ycenter), (0, 0, 0), 1)
      cv2.line(full_grid_image, (0, ycenter), (ww-1, ycenter), (0, 0, 0), 1)
  # Return results
  return {
    'threshold': thresh,
    'vertical': vertical_img,
    'horizontal': horizontal_img,
    'full': full_grid_image
  }
