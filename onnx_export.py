# import onnx

# model = onnx.load("yolov8n.onnx")
# print(onnx.helper.printable_graph(model.graph))
from ultralytics import YOLO

# Load a pretrained or custom model
model = YOLO('yolov8n.pt')

# Export to ONNX format with dynamic input support
model.export(format='onnx', dynamic=True, imgsz=640,save_dir='model')  # default imgsz is 640

# Load the exported ONNX model
onnx_model = YOLO("yolo8vn.onnx")

# Run inference
results = onnx_model("https://ultralytics.com/images/bus.jpg")