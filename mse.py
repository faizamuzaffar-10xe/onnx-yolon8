import numpy as np

import cv2 
import onnxruntime as ort
import numpy as np
import numpy

# 🔧 STEP 1: Preprocess the input image

def preprocess_image(img_path, input_size=(640, 640)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # CHW → BCHW
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    print("image",img.shape, img.dtype)

    return img


def get_constant(model_input_shape):
  strides = [8, 16, 32]
  temp1 = [model_input_shape[1]/x for x in strides]
  temp2 = [model_input_shape[2]/x for x in strides]
  temp3 = []
  temp4 = [] 
  
  for i in range(len(temp1)):
    for j in numpy.arange(0.5, temp1[i], 1):
      for k in numpy.arange(0.5, temp2[i], 1):
        temp3.append(k)
        temp4.append(j)

  constant = numpy.array([temp3, temp4])
  return numpy.expand_dims(constant, axis = 0)

def get_constant_pose():
  strides = [80, 40, 20]
  temp = [80, 40, 20]
  temp1 = []
  temp2 = []
  for i in range(len(strides)):
    for j in numpy.arange(0, strides[i], 1):
      for k in numpy.arange(0, temp[i], 1):
        temp1.append(k)
        temp2.append(j)

  constant = numpy.array([temp1, temp2])
  return numpy.expand_dims(constant, axis = 0)

def get_mul_constant(model_input_shape):
  strides = [8, 16, 32]
  temp1 = [model_input_shape[1]/x for x in strides]
  temp2 = [model_input_shape[2]/x for x in strides]
  strides_space = [int(x*y) for x,y in zip(temp1, temp2)]  # for 6300 output dimension model
  constant = []
  
  for ranges, stride in zip(strides_space, strides):
    for i in range(ranges):
      constant.append(stride)

  return numpy.array([constant])
# Load preprocessed image
img_input = preprocess_image("bus.jpg")  # Replace with your image path

# 🔧 STEP 2: Load first subgraph and run inference
# session = ort.InferenceSession("yolov8n_first_subgraph.onnx")
try:
    session = ort.InferenceSession("subgraph1.onnx")
    print("Session created successfully.")
except Exception as e:
    print("Failed to create session:", e)

# # Get input name for the session (usually 'images' or similar)
input_name = session.get_inputs()[0].name
print(f"Input name: {input_name}")

# Run inference on the preprocessed image
intermediate_outputs = session.run(None, {input_name: img_input})

# # 🔍 STEP 3: Print output shapes and types
# for i, output in enumerate(intermediate_outputs):
#     print(f"Intermediate Output {i} shape: {output.shape}, dtype: {output.dtype}")

# Optional: Save outputs to pass to second subgraph
Concat_output_0 = intermediate_outputs[0]
Conv_output_0 = intermediate_outputs[1]
sigmoid_output = intermediate_outputs[2]
#print("Concat_output_0 type,dtype",Concat_output_0.dtype,type(Concat_output_0))

print(Concat_output_0.shape,Conv_output_0.shape,sigmoid_output.shape)







# Step 1: Reshape inputs
x1 = Concat_output_0.reshape(1, 17, 3, 8400)     # shape: 1×17×3×8400
x2 = Conv_output_0.reshape(1, 4, 8400)           # shape: 1×4×8400

# Step 2: Slice operations
# Placeholder slicing indices - replace with actual ones
slice1_concat_l = x1[:, :, :2, :]                         # shape: 1×17×2×8400
slice2_concat_r = x1[:, :, 2:3, :]                         # shape: 1×17×3×8400 (no-op for completeness)
slice3_conv_l = x2[:, :2, :]                            # shape: 1×2×8400
slice4_conv_r = x2[:, 2:4, :]                            # shape: 1×2×8400

# Step 3: Arithmetic Operations
x3_concat_l = slice1_concat_l * 2                                  # shape: 1×17×2×8400
# x4 = x3 + np.ones((1, 2, 8400)) * 2              # Broadcasting Add, shape: 1×17×2×8400
# x5 = x4 * np.ones((1, 2, 8400))                  # Broadcasting Mul, shape: 1×17×2×8400
x4_concat_l = x3_concat_l + get_constant_pose()

print("x3_concat_l shape",x3_concat_l.shape,get_constant_pose().shape)

x5_concat_l = x4_concat_l *  get_mul_constant((3,640,640))
print("x4_concat_l shape",x4_concat_l.shape,get_mul_constant((3,640,640)).shape)
# Slice for Sigmoid
x6_concat_r = slice2_concat_r                           # shape: 1×17×1×8400
x7_concat_r = 1 / (1 + np.exp(-x6_concat_r))                       # Sigmoid activation

# Concatenate: shape becomes 1×17×3×8400
x_concat1 = np.concatenate([x5_concat_l, x7_concat_r], axis=2)

# Reshape for final concat
x_concat1_reshaped = x_concat1.reshape(1, 51, 8400)

# sub 1 sub 2
add1 = slice4_conv_r + get_constant((3,640,640)) 
sub1 = slice3_conv_l - get_constant((3,640,640))
print("slice4_conv_r shape",slice4_conv_r.shape,get_constant((3,640,640)).shape)
print("slice3_conv_l shape",slice3_conv_l.shape,get_constant((3,640,640)).shape)
# Right branch: arithmetic on x2 slices
x8_conv_sub = add1 - sub1                     # shape: 1×2×8400
x9_conv_add = sub1 + add1                             # shape: 1×2×8400



x10_conv_r = x9_conv_add / 2     #add                                # shape: 1×2×8400



# Concat -> shape: 1×4×8400
x_concat2_conv = np.concatenate([x8_conv_sub, x10_conv_r], axis=1)
#x11 = x_concat2 * np.ones((1, 4, 8400))          # Broadcasting Mul
x11_conv = x_concat2_conv * get_mul_constant((3,640,640)) 
print("x_concat2_conv shape",x_concat2_conv.shape,get_mul_constant((3,640,640)).shape)  
# Final concat -> shape: 1×56×8400
final_concat = np.concatenate([x_concat1_reshaped, x11_conv,sigmoid_output], axis=1)

# Final output
output0 = final_concat  # shape: 1×56×8400

print("out shape",output0.shape)


#subgraph 1, and subgraph 2


############
###############

####################
# variable comparison

import onnx
from onnx import utils

# Load original ONNX model
onnx_model = onnx.load("yolov8n-pose.onnx")
input_names = [input.name for input in onnx_model.graph.input]


def mse(output_raw,session,sectionx):
    # Run inference on the preprocessed image
    intermediate_outputs = session.run(None, {input_name: img_input})


    output_rawx = [output_raw, x6_concat_r, x8_conv_sub, x10_conv_r]

    for i, item in enumerate(intermediate_outputs[0:-1]):


        mse = np.mean((item - output_rawx[i]) ** 2)

        print(sectionx,mse)


intermediate_add = [

    "/model.22/Add_3_output_0",
    "/model.22/Slice_3_output_0",
    "/model.22/Sub_1_output_0",
    "/model.22/Div_1_output_0",
    "/model.22/Sigmoid_output_0"
]
intermediate_mul = [
    "/model.22/Mul_3_output_0",
    "/model.22/Slice_3_output_0",
    "/model.22/Sub_1_output_0",
    "/model.22/Div_1_output_0",
    "/model.22/Sigmoid_output_0"
]
intermediate_slice = [

    "/model.22/Slice_2_output_0",
    "/model.22/Slice_3_output_0",
    "/model.22/Slice_output_0",
    "/model.22/Slice_1_output_0",
    "/model.22/Sigmoid_output_0"
]


intermediate_concat = [

    "/model.22/Concat_6_output_0",
    "/model.22/Concat_5_output_0",
    "/model.22/Sigmoid_output_0"
]


intermediate_add_sub = [

    "/model.22/Concat_6_output_0",
    "/model.22/Sub_output_0",
    "/model.22/Add_1_output_0",
    "/model.22/Sigmoid_output_0"
]

# The final outputs (these remain the same)



utils.extract_model(
    input_path="yolov8n-pose.onnx",
    output_path="slice_output.onnx",
    input_names=input_names,
    output_names=intermediate_slice
)
utils.extract_model(
    input_path="yolov8n-pose.onnx",
    output_path="add_output.onnx",
    input_names=input_names,
    output_names=intermediate_add
)
utils.extract_model(
    input_path="yolov8n-pose.onnx",
    output_path="mul_output.onnx",
    input_names=input_names,
    output_names=intermediate_mul
)
utils.extract_model(
    input_path="yolov8n-pose.onnx",
    output_path="concat_output.onnx",
    input_names=input_names,
    output_names=intermediate_concat
)


utils.extract_model(
    input_path="yolov8n-pose.onnx",
    output_path="add_sub.onnx",
    input_names=input_names,
    output_names=intermediate_add_sub
)

'''slice1_concat_l = x1[:, :, :2, :]                         # shape: 1×17×2×8400
slice2_concat_r = x1[:, :, 2:3, :]                         # shape: 1×17×3×8400 (no-op for completeness)
slice3_conv_l = x2[:, :2, :]                            # shape: 1×2×8400
slice4_conv_r = x2[:, 2:4, :]    '''
img_input = preprocess_image("bus.jpg")  # Replace with your image path

session_slice = ort.InferenceSession("slice_output.onnx")

# for slice

input_name = session_slice.get_inputs()[0].name
print(f"Input name: {input_name}")
#mse([slice1_concat_l, slice2_concat_r, slice3_conv_l,slice1_concat_l, slice4_conv_r,],session_slice,"slice")


# Run inference on the preprocessed image

intermediate_outputs_slice = session_slice.run(None, {input_name: img_input})

print("Slice",np.mean((intermediate_outputs_slice[0] - slice1_concat_l) ** 2))
print("Slice",np.mean((intermediate_outputs_slice[1] - slice2_concat_r) ** 2))
print("Slice",np.mean((intermediate_outputs_slice[2] - slice3_conv_l) ** 2))
print("Slice",np.mean((intermediate_outputs_slice[3] - slice4_conv_r) ** 2))

#################################################################################################



session_add = ort.InferenceSession("add_output.onnx")

# for add

input_name = session_add.get_inputs()[0].name
print(f"Input name: {input_name}")

# Run inference on the preprocessed image
intermediate_outputs_add = session_add.run(None, {input_name: img_input})


#mse([x4_concat_l, x6_concat_r, x8_conv_sub, x10_conv_r],session_add,"add")

print("Add",np.mean((intermediate_outputs_add[0] - x4_concat_l) ** 2))

# ###################################################################################

session_mul = ort.InferenceSession("mul_output.onnx")


input_name = session_mul.get_inputs()[0].name
print(f"Input name: {input_name}")

# Run inference on the preprocessed image
intermediate_mul = session_mul.run(None, {input_name: img_input})

#mse([x3_concat_l, x6_concat_r, x8_conv_sub, x10_conv_r],session_mul,"mul")



print(np.mean((intermediate_mul[0] - x3_concat_l) ** 2))


############################################################################

session_concat = ort.InferenceSession("concat_output.onnx")


input_name = session_concat.get_inputs()[0].name
print(f"Input name: {input_name}")

# Run inference on the preprocessed image
intermediate_concat = session_concat.run(None, {input_name: img_input})

#mse([x3_concat_l, x6_concat_r, x8_conv_sub, x10_conv_r],session_mul,"mul")



print(np.mean((intermediate_concat[0] - x_concat1) ** 2))


print(np.mean((intermediate_concat[1] - x_concat2_conv) ** 2))



#############################

session_add_sub = ort.InferenceSession("add_output.onnx")

# for add

input_name = session_add_sub.get_inputs()[0].name
print(f"Input name: {input_name}")

# Run inference on the preprocessed image
intermediate_add_sub = session_add_sub.run(None, {input_name: img_input})


#mse([x4_concat_l, x6_concat_r, x8_conv_sub, x10_conv_r],session_add,"add")

print(np.mean((intermediate_add_sub[0] - add1) ** 2))

print(np.mean((intermediate_add_sub[1] - sub1) ** 2))


