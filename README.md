# YOLOv9 Object Detection Example
![Inference Image](https://github.com/Brianhulela/yolov9_object_detection/blob/main/bounding_boxes_result.jpg)

This project demonstrates how to use a YOLOv9 model to perform object detection on an image using the `ultralytics`, `cv2`, and `supervision` libraries. The model detects objects in a scene and annotates them with bounding boxes.

For a step-by-step guide checkout [this article](https://hulela.co.za/run-yolov9-object-detection-model-locally-43c23c7078b4)

## Installation

Before running the script, you'll need to install the following dependencies:

```
pip install ultralytics supervision
```
Alternatively you can install the dependecies using the requirements.txt file:

```
pip install -r requirements.txt
```

Additionally, you'll need the YOLOv9 pretrained model weights (`yolov9t.pt`), which can be downloaded from the [Ultralytics YOLO repository](https://github.com/ultralytics/yolov5/releaseshttps://docs.ultralytics.com/models/yolov9/).

## Code Explanation

1. **Image Download**: The image is downloaded using `urllib`.
2. **Model Loading**: A YOLOv9 model is loaded from pretrained weights.
3. **Inference**: The model runs inference on the image to detect objects.
4. **Bounding Box Annotation**: Detected objects are annotated with bounding boxes.
5. **Image Saving**: The annotated image is saved to the disk.

## How to Run

1. Install the required dependencies listed in the installation section.
3. Run the provided Python script.

## Output

The annotated image will be saved as `bounding_boxes_result.jpg` in the current working directory.
