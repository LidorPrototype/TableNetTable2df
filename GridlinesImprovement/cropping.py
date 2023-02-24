import cv2
import numpy as np


def cropImage(old_image: np.ndarray) -> np.ndarray:
    # read image
    img = old_image
    hh, ww = img.shape[:2]
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # threshold
    thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
    # crop 1 pixel and add 1 pixel white border to ensure outer white regions not considered small contours
    thresh = thresh[1:hh-1, 1:ww-1]
    thresh = cv2.copyMakeBorder(thresh, 1,1,1,1, borderType=cv2.BORDER_CONSTANT, value=(255,255,255))
    # get contours
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    # get min and max x and y from all bounding boxes larger than half the image size
    area_thresh = hh * ww / 2
    xmin = ww
    ymin = hh
    xmax = 0
    ymax = 0
    for cntr in contours:
        area = cv2.contourArea(cntr)
        if area < area_thresh:
            x,y,w,h = cv2.boundingRect(cntr)
            xmin = x if (x < xmin) else xmin
            ymin = y if (y < ymin) else ymin
            xmax = x+w-1 if (x+w-1 > xmax ) else xmax
            ymax = y+h-1 if (y+h-1 > ymax) else ymax
    # draw bounding box     
    bbox = img.copy()
    cv2.rectangle(bbox, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    # crop img at bounding box, but add 2 all around to keep the black lines
    result = img[ymin:ymax, xmin:xmax]
    return result

