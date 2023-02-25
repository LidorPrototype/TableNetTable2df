import numpy as np
import cv2


def removeLines(old_image: np.ndarray, axis) -> np.ndarray:
    gray = cv2.cvtColor(old_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if axis == "horizontal":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    elif axis == "vertical":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    else:
        raise ValueError("Axis must be either 'horizontal' or 'vertical' in order to work")
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
    contours = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    result = old_image.copy()
    for contour in contours:
        cv2.drawContours(result, [contour], -1, (255, 255, 255), 2)
    return result
