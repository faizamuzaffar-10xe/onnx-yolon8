import onnx
import onnx_graphsurgeon as gs

# Load model
onnx_model = onnx.load("yolov8n.onnx")
graph = gs.import_onnx(onnx_model)

# Replace these names with actual ones from Netron
intermediate_output_names = [
    "output1",
    "/model.22/Concat_output_0",
    "/model.22/dfl/conv/Conv_output_0",
    "/model.22/Sigmoid_output_0"]

# Mark these tensors as outputs
intermediate_outputs = [tensor for tensor in graph.tensors().values() if tensor.name in intermediate_output_names]
graph.outputs = intermediate_outputs

# Export subgraph1
onnx_subgraph1 = gs.export_onnx(graph)
onnx.save(onnx_subgraph1, "subgraph1.onnx")





# Load original full graph again
graph = gs.import_onnx(onnx.load("yolov8n.onnx"))

# Set intermediate nodes as new inputs
intermediate_inputs = [tensor for tensor in graph.tensors().values() if tensor.name in intermediate_output_names]
graph.inputs = intermediate_inputs

# Set original output node(s) as output
final_outputs = [tensor for tensor in graph.outputs]
graph.outputs = final_outputs

# Export subgraph2
onnx_subgraph2 = gs.export_onnx(graph)
onnx.save(onnx_subgraph2, "subgraph2.onnx")
