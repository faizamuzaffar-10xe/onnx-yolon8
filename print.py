# import onnx
import onnxruntime as ort
import onnx

# Load the ONNX model
onnx_model = onnx.load('yolov8n.onnx')
model= onnx.load('yolov8n.onnx')
for out in model.graph.output:
    print(out.name)
