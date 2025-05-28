import onnxruntime as ort
import numpy as np
import cv2

# Load image and preprocess
img = cv2.imread("bus.jpg")
img = cv2.resize(img, (640, 640))
img = img.transpose(2, 0, 1)  # HWC to CHW
img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

# Load ONNX model
session = ort.InferenceSession("yolov8n.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Run inference
outputs = session.run(None, {input_name: img})
print(outputs)
