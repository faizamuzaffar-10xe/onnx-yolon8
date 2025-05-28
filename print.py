# import onnx
import onnxruntime as ort
import onnx

# Load the ONNX model
onnx_model = onnx.load('yolov8n.onnx')

# # Print model metadata
# print(f"Model IR version: {onnx_model.ir_version}")
# print(f"Producer name: {onnx_model.producer_name}")
# print(f"Producer version: {onnx_model.producer_version}")

# # Create ONNX Runtime session
# sess = ort.InferenceSession('yolov8n.onnx')

# # Print input information
# print("\nInputs:")
# for i, input in enumerate(sess.get_inputs()):
#     print(f"  Input {i}:")
#     print(f"    Name: {input.name}")
#     print(f"    Shape: {input.shape}")
#     print(f"    Type: {input.type}")

# # Print output information
# print("\nOutputs:")
# for i, output in enumerate(sess.get_outputs()):
#     print(f"  Output {i}:")
#     print(f"    Name: {output.name}")
#     print(f"    Shape: {output.shape}")
#     print(f"    Type: {output.type}")
model= onnx.load('yolov8n.onnx')
for out in model.graph.output:
    print(out.name)