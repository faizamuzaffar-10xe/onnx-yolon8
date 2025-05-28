
# from ultralytics import YOLO

# # Load your trained model or a pretrained one
# model = YOLO('yolov8n.pt')  # or 'path/to/your/model.pt'

# # Export to ONNX format
# model.export(format='onnx', dynamic=True)


# # Load the exported ONNX model
# onnx_model = YOLO("yolov8n.onnx")

# # Run inference
# results = onnx_model("https://ultralytics.com/images/bus.jpg")




import onnx

model = onnx.load("yolov8n.onnx")
print(onnx.helper.printable_graph(model.graph))