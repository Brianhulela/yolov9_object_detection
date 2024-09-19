from ultralytics import YOLO
import cv2
import urllib.request
import supervision as sv
import numpy as np

url, filename = ("https://plus.unsplash.com/premium_photo-1661713745988-b0945c12efd1?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MjI1fHxzdHJlZXQlMjBwZW9wbGV8ZW58MHx8MHx8fDA%3D", "scene.jpg")
urllib.request.urlretrieve(url, filename)

# Read an image
image = cv2.imread(filename)

# Build a YOLOv9c model from pretrained weight
model = YOLO("yolov9t.pt")

# Run inference with the YOLOv9t model on the downloaded image
results = model(image)

# Extract bounding boxes, classes, names, and confidences
boxes = results[0].boxes.xyxy.tolist()
classes = results[0].boxes.cls.tolist()
classes = [int(cls) for cls in classes]

names = results[0].names
confidences = results[0].boxes.conf.tolist()

detections = sv.Detections(
  xyxy=np.array(boxes),
  class_id=np.array(classes),
  confidence=np.array(confidences),
)

bounding_box_annotator = sv.BoundingBoxAnnotator()
annotated_frame = bounding_box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

sv.plot_image(annotated_frame)

# Save the annotated image
cv2.imwrite("bounding_boxes.jpg", annotated_frame)