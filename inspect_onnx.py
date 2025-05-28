# import onnx

# # Load the model
# model = onnx.load('yolov8n.onnx')

# # Print model inputs
# print("Inputs:")
# for input in model.graph.input:
#     shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
#     dtype = input.type.tensor_type.elem_type
#     print(f"  Name: {input.name}, Shape: {shape}, Type: {dtype}")

# # Print model outputs
# print("Outputs:")
# for output in model.graph.output:
#     shape = [dim.dim_value for dim in output.type.tensor_type.shape.dim]
#     dtype = output.type.tensor_type.elem_type
#     print(f"  Name: {output.name}, Shape: {shape}, Type: {dtype}")


# import onnx

# # Load the ONNX model
# model = onnx.load("yolov8n.onnx")
# graph = model.graph

# print("Inputs:")
# for input in graph.input:
#     name = input.name
#     shape = [dim.dim_value if dim.HasField("dim_value") else "dynamic" for dim in input.type.tensor_type.shape.dim]
#     dtype = input.type.tensor_type.elem_type
#     print(f"  Name: {name}, Shape: {shape}, Type: {dtype}")

# print("\nOutputs:")
# for output in graph.output:
#     name = output.name
#     shape = [dim.dim_value if dim.HasField("dim_value") else "dynamic" for dim in output.type.tensor_type.shape.dim]
#     dtype = output.type.tensor_type.elem_type
#     print(f"  Name: {name}, Shape: {shape}, Type: {dtype}")


import onnx

# Load ONNX model
#model_path = 'yolov8n-pose.onnx'
model = onnx.load("yolov8n.onnx")

# Print all intermediate tensor names
print("All intermediate tensor names:")
for node in model.graph.node:
    for output in node.output:
        print(output)
        print(output.shape,output.dtype)

