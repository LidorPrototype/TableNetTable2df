import cv2
import numpy as np


def cropImage(old_image: np.ndarray) -> np.ndarray:
    # Get dimensions
    hh, ww = old_image.shape[:2]
    # Convert to gray
    gray = cv2.cvtColor(old_image, cv2.COLOR_BGR2GRAY)
    # Threshold
    thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]
    # Crop 1 pixel and add 1 pixel white border to ensure outer white regions not considered the small contours
    thresh = thresh[1: hh - 1, 1 : ww - 1]
    thresh = cv2.copyMakeBorder(thresh, 1, 1, 1, 1, borderType = cv2.BORDER_CONSTANT, value = (255, 255, 255))
    # Get contours
    contours = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # Get min and max x and y from all bounding boxes larger than half of the image size
    thresh_area = hh * ww / 2
    xmin = ww
    ymin = hh
    xmax = 0
    ymax = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < thresh_area:
            x, y, w, h = cv2.boundingRect(contour)
            xmin = x if (x < xmin) else xmin
            ymin = y if (y < ymin) else ymin
            xmax = x + w - 1 if (x + w - 1 > xmax ) else xmax
            ymax = y + h - 1 if (y + h - 1 > ymax) else ymax
    # Draw bounding box     
    bounding_box = old_image.copy()
    cv2.rectangle(bounding_box, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    # Crop old_image at the bounding box, but add 2 all around to keep the black lines
    result = old_image[ymin : ymax, xmin : xmax]
    return result
