import onnx
from onnx import utils

# Load original ONNX model
onnx_model = onnx.load("yolov8n-pose.onnx")
input_names = [input.name for input in onnx_model.graph.input]

# Define the intermediate outputs that will serve as the *inputs* to subgraph2
intermediate_outputs = [
    "/model.22/Concat_output_0",
    "/model.22/dfl/conv/Conv_output_0",
    "/model.22/Sigmoid_output_0"
]

# The final outputs (these remain the same)

final_outputs = [output.name for output in onnx_model.graph.output]
print(final_outputs)

utils.extract_model(
    input_path="yolov8n-pose.onnx",
    output_path="subgraph1.onnx",
    input_names=input_names,
    output_names=intermediate_outputs
)

# Extract second subgraph: from intermediate to final outputs
utils.extract_model(
    input_path="yolov8n-pose.onnx",
    output_path="subgraph2.onnx",
    input_names=intermediate_outputs,
    output_names=['output0']
)
# Extract second subgraph: from intermediate to final outputs
utils.extract_model(
    input_path="yolov8n-pose.onnx",
    output_path="subgraph3.onnx",
    input_names=input_names,
    output_names=['output0']
)
