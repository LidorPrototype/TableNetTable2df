import cv2
import matplotlib.pyplot as plt
from GridlinesImprovement.draw_gridlines_functions import getHorizontalCnt, getVerticalCnt, drawHorizontalLines, drawVerticalLines
from GridlinesImprovement.remove_gridlines import removeLines


image_path = "Model Implementation/DummyDatabase/test_images/image_gridless.png"
new_image_path = "Model Implementation/DummyDatabase/test_images/image_grided.png"
original = cv2.imread(image_path)
# Remove all gridlines
gridless = removeLines(removeLines(original, 'horizontal'), 'vertical')
# Get vertical lines
vertical_cnt, hh_ = getVerticalCnt(gridless.copy())
# Get horizontal lines
horizontal_cnt, ww_ = getHorizontalCnt(gridless.copy())
# Draw horizontal lines
horizontal_only = drawHorizontalLines(gridless.copy(), horizontal_cnt, ww_)
# Draw vertical lines
full_grid = drawVerticalLines(horizontal_only.copy(), vertical_cnt, hh_)
# Save new image
cv2.imwrite(new_image_path, full_grid)
# Show image
plt.imshow(full_grid)
