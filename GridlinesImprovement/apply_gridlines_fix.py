import cv2
import matplotlib.pyplot as plt
from GridlinesImprovement.draw_gridlines_functions import drawGridlines
from GridlinesImprovement.cropping import cropImage
from GridlinesImprovement.remove_gridlines import removeLines


image_path = "Model Implementation/DummyDatabase/test_images/image_gridless.png"
new_image_path = "Model Implementation/DummyDatabase/test_images/image_grided.png"
original = cv2.imread(image_path)
# Remove all gridlines
gridless = removeLines(removeLines(original, 'horizontal'), 'vertical')
# Draw grid lines
images_by_stage = drawGridlines(gridless.copy())
"""
    images_by_stage: (dict)
        'threshold': threshold image
        'vertical': vertical grid lines image
        'horizontal': horizontal grid lines image
        'full': full grid lines image
"""
# Obtain full grid image
full_image = images_by_stage['full'].copy()
# Crop image
cropped_image = cropImage(full_image.copy())
# Save new image
cv2.imwrite(new_image_path, cropped_image)
# Show image
plt.imshow(cropped_image)