import cv2
import numpy as np

# Load the image of the solar panel
image = cv2.imread(r"C:\Users\Raghav Prabanth\OneDrive\Desktop\PROJECTS RELATED FILES\python learner files\solarmicrocracks.png")

# Resize image for better visualization (optional)
image = cv2.resize(image, (800, 600))

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Improve contrast using CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_gray = clahe.apply(gray_image)

# 2. Apply GaussianBlur to reduce noise before further processing
blurred_image = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)

# 3. Apply adaptive thresholding for better detection
adaptive_thresh = cv2.adaptiveThreshold(
    blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# 4. Morphological operations for both dust and crack detection
kernel_dust = np.ones((3, 3), np.uint8)  # Kernel for dust
kernel_crack = np.ones((5, 5), np.uint8)  # Larger kernel for cracks

# Apply morphological operations
morph_close_dust = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_dust)
morph_close_crack = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel_crack)

# 5. Edge detection for cracks
edges_crack = cv2.Canny(morph_close_crack, 50, 150)

# 6. Detect contours for cracks
contours_crack, _ = cv2.findContours(edges_crack, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 7. Edge detection for dust (small particles)
edges_dust = cv2.Canny(morph_close_dust, 30, 100)

# 8. Detect contours for dust
contours_dust, _ = cv2.findContours(edges_dust, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Copy original image to draw contours
detection_output = image.copy()

# 9. Filter and draw contours for cracks
for contour in contours_crack:
    area = cv2.contourArea(contour)
    if area > 50:  # Threshold for major cracks
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 0.2 < aspect_ratio < 5.0:  # Aspect ratio for cracks
            cv2.drawContours(detection_output, [contour], -1, (0, 255, 0), 2)  # Green for cracks

# 10. Filter and draw contours for dust
for contour in contours_dust:
    area = cv2.contourArea(contour)
    if 5 < area <= 50:  # Threshold for dust particles
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if 0.2 < aspect_ratio < 10.0:  # Aspect ratio for dust particles
            cv2.drawContours(detection_output, [contour], -1, (255, 0, 0), 1)  # Blue for dust particles

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Grayscale', enhanced_gray)
cv2.imshow('Adaptive Threshold', adaptive_thresh)
cv2.imshow('Dust Detection', edges_dust)
cv2.imshow('Crack Detection', edges_crack)
cv2.imshow('Detected Dust and Cracks', detection_output)

# Save the results (optional)
cv2.imwrite('dust_and_crack_detection_output.jpg', detection_output)

# Wait for key press and close windows
cv2.waitKey(0)
cv2.destroyAllWindows()
