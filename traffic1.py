import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image (replace 'your_image.jpg' with your actual image file)
image_path = 'image3.jpeg'
image = cv2.imread(image_path)

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the yellow color range (for traffic lights)
lower_yellow = np.array([20, 0, 0])
upper_yellow = np.array([40, 255, 255])

# Create a mask for yellow regions
mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Apply median blur to reduce noise
blurred_mask = cv2.medianBlur(mask, 5)

# Find contours in the blurred mask
contours, _ = cv2.findContours(blurred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw rectangles around detected traffic lights
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Now let's detect poles (vertical lines)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray_image, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

# Display the result
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Traffic Lights and Poles")
plt.axis("off")
plt.show()
