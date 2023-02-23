import cv2


def getHorizontalCnt(old_image):
  # read image
  img = old_image.copy() # cv2.imread(image_path1)
  hh, ww = img.shape[:2]
  # convert to grayscale 
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # average gray image to one row
  row = cv2.resize(gray, (1,hh), interpolation = cv2.INTER_AREA)
  # threshold on white
  thresh = cv2.threshold(row, 250, 255, cv2.THRESH_BINARY_INV)[1]
  # get contours
  contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1]
  return contours, ww


def getVerticalCnt(old_image):
  # read image
  img = old_image.copy() # cv2.imread(image_path1)
  hh, ww = img.shape[:2]
  # convert to grayscale 
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # average gray image to one column
  column = cv2.resize(gray, (ww, 1), interpolation = cv2.INTER_AREA)
  # threshold on white
  thresh = cv2.threshold(column, 254, 255, cv2.THRESH_BINARY)[1]
  # get contours
  contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours = contours[0] if len(contours) == 2 else contours[1]
  return contours, hh


def drawHorizontalLines(old_image, contours_h, ww_):
    image = old_image.copy()
    for cntr in contours_h:
        x, y, w, h = cv2.boundingRect(cntr)
        ycenter = y + h // 2
        cv2.line(image, (0, ycenter), (ww_ - 1, ycenter), (0, 0, 0), 1)
    return image


def drawVerticalLines(old_image, contours_v, hh_):
    image = old_image.copy()
    for cntr in contours_v:
        x, y, w, h = cv2.boundingRect(cntr)
        xcenter = x + w // 2
        cv2.line(image, (xcenter, 0), (xcenter, hh_ - 1), (0, 0, 0), 1)
    return image
