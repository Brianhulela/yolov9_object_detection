from ultralytics import YOLO
import cv2
import urllib.request
import random

url, filename = ("https://plus.unsplash.com/premium_photo-1661713745988-b0945c12efd1?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjI1fHxzdHJlZXQlMjBwZW9wbGV8ZW58MHx8MHx8fDA%3D", "scene.jpg")
urllib.request.urlretrieve(url, filename)

# Read an image
image = cv2.imread(filename)

# Predefined list of distinct bright colors (in RGB format)
bright_colors = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 165, 0),   # Orange
    (0, 255, 255),   # Cyan
    (255, 0, 255),   # Magenta
    (75, 0, 130),    # Indigo
    (238, 130, 238), # Violet
    (0, 128, 255),   # Light Blue
]

# Function to randomly select a bright color from the list
def get_random_color():
    return random.choice(bright_colors)

# Build a YOLOv9c model from pretrained weight
model = YOLO("yolov9t.pt")

# Run inference with the YOLOv9t model on the downloaded image
results = model(image)

# Extract bounding boxes, classes, names, and confidences
boxes = results[0].boxes.xyxy.tolist()
classes = results[0].boxes.cls.tolist()
names = results[0].names
confidences = results[0].boxes.conf.tolist()

# Create a dictionary to hold colors for each class
class_color_map = {}

# Iterate through the results
for box, cls, conf in zip(boxes, classes, confidences):
    x0, y0, x1, y1 = box
    name = names[int(cls)]

    # If this class doesn't have a color, generate a random one and store it
    if int(cls) not in class_color_map:
        class_color_map[int(cls)] = get_random_color()

    # Get the color for this class
    color = class_color_map[int(cls)]

    # Draw the bounding box with the color
    start_point = (int(x0), int(y0))
    end_point = (int(x1), int(y1))
    cv2.rectangle(image, start_point, end_point, color=color, thickness=2)
    
    # Draw the label text with the same color
    cv2.putText(
        image,
        f"{name} {conf:.2f}",
        (int(x0), int(y0) - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=color,
        thickness=2
    )

# Save the annotated image
cv2.imwrite("example_with_bounding_boxes.jpg", image)