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
print(Concat_output_0.dtype,type(Concat_output_0))









# Step 1: Reshape inputs
x1 = Concat_output_0.reshape(1, 17, 3, 8400)     # shape: 1×17×3×8400
x2 = Conv_output_0.reshape(1, 4, 8400)           # shape: 1×4×8400

# Step 2: Slice operations
# Placeholder slicing indices - replace with actual ones
slice1 = x1[:, :, :2, :]                         # shape: 1×17×2×8400
slice2 = x1[:, :, 2:3, :]                          # shape: 1×17×3×8400 (no-op for completeness)
slice3 = x2[:, :2, :]                            # shape: 1×2×8400
slice4 = x2[:, 2:4, :]                            # shape: 1×2×8400

# Step 3: Arithmetic Operations
x3 = slice1 * 2                                  # shape: 1×17×2×8400
# x4 = x3 + np.ones((1, 2, 8400)) * 2              # Broadcasting Add, shape: 1×17×2×8400
# x5 = x4 * np.ones((1, 2, 8400))                  # Broadcasting Mul, shape: 1×17×2×8400
x4 = x3 + get_constant_pose()
val_get_constant_pose=get_constant_pose()
x5 = x4 *  get_mul_constant((3,640,640))
# Slice for Sigmoid
x6 = slice2                             # shape: 1×17×1×8400
x7 = 1 / (1 + np.exp(-x6))                       # Sigmoid activation

# Concatenate: shape becomes 1×17×3×8400
x_concat1 = np.concatenate([x5, x7], axis=2)

# Reshape for final concat
x_concat1_reshaped = x_concat1.reshape(1, 51, 8400)


# sub 1 sub 2
sub1 = slice3 + get_constant((3,640,640)) 
add1 = slice4 - get_constant((3,640,640))

val_get_constant=get_constant((3,640,640))
# Right branch: arithmetic on x2 slices
x8 = sub1 - add1                       # shape: 1×2×8400
x9 = sub1 + add1                             # shape: 1×2×8400

x10 = x9 / 2                                     # shape: 1×2×8400

# Concat -> shape: 1×4×8400
x_concat2 = np.concatenate([x8, x10], axis=1)
#x11 = x_concat2 * np.ones((1, 4, 8400))          # Broadcasting Mul
x11 = x_concat2 * get_mul_constant((3,640,640))   
val_get_mul_constant=get_mul_constant((3,640,640))  
# Final concat -> shape: 1×56×8400
final_concat = np.concatenate([x_concat1_reshaped, x11,sigmoid_output], axis=1)

# Final output
output0 = final_concat  # shape: 1×56×8400

print(output0.shape,"out shape")


sub = np.load("./variables/sub.npy")
mul = np.load("variables/mul.npy")
add_conv = np.load("variables/add_conv.npy")
add_concat = np.load("variables/add_concat.npy")

print("Loaded array shape:", sub.shape)
print("Data type:", sub.dtype)

# val_get_constant
# val_get_constant_pose

# val_get_mul_constant
mse1 = np.mean((val_get_constant- sub) ** 2)

mse2 = np.mean((mul- val_get_mul_constant) ** 2)


mse3 = np.mean((add_concat-val_get_constant_pose ) ** 2)


mse4 = np.mean((add_concat-val_get_constant_pose ) ** 2)
print(mse1,mse2,mse3,mse4)

# print(mse)
# mse1=np.mean()