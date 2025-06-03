

#For video boxes and pose full
import cv2
import numpy as np
import numpy
import onnxruntime as ort

# --- Constants
video_path = "Trololo [iwGFalTRHDA].mp4"
onnx_path = "subgraph1.onnx"
input_size = (640, 640)
conf_threshold = 0.6
nms_threshold = 0.5

# --- Session
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name

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
# def nms(dets, iou_thresh=0.5):
#     if len(dets) == 0:
#         return []
#     boxes = np.array([d[:4] for d in dets])
#     scores = np.array([d[4] for d in dets])
#     indices = cv2.dnn.NMSBoxes(
#         bboxes=boxes.tolist(),
#         scores=scores.tolist(),
#         score_threshold=0.4,
#         nms_threshold=iou_thresh
#     )
#     if len(indices) == 0:
#         return []
#     if isinstance(indices, np.ndarray):
#         indices = indices.flatten()
#     return [dets[i] for i in indices]


def nms(dets, iou_thresh=0.5):
        if len(dets) == 0:
            return []

        dets = np.array(dets)
        boxes = dets[:, :4]
        scores = dets[:, 4]

        # Convert (x1, y1, x2, y2)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]  # Sort by score descending

        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(dets[i])

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0.0, xx2 - xx1 + 1)
            inter_h = np.maximum(0.0, yy2 - yy1 + 1)
            inter_area = inter_w * inter_h

            union_area = areas[i] + areas[order[1:]] - inter_area
            iou = inter_area / np.maximum(union_area, 1e-6)

            inds = np.where(iou <= iou_thresh)[0]
            order = order[inds + 1]

        return keep

# Post processing Function
def xywh_to_xyxy_np(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)

def draw_detections(img, detections):
    for det in detections:
        x1, y1, x2, y2, score, class_id = det[:6]
        if int(class_id) != 0:
            continue  # Only show person (class ID 0)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"Person: {score * 100:.1f}%"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Person : {str(class_scores[0][0])[:6]}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img

# Draw poses
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0

def draw_pose(img, dets, output, mask):
    COCO_PAIRS = [
        (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (5, 6), (11, 12), (5, 11), (6, 12),
        (0, 1), (1, 3), (0, 2), (2, 4),
        (0, 5), (0, 6)
    ]
    for det in dets:
        best_iou = 0
        best_idx = None
        for i, box in enumerate(filtered_boxes_scaled):
            iou_score = iou(det[:4], xywh_to_xyxy_np(np.expand_dims(box, axis=0))[0])
            if iou_score > best_iou and iou_score > 0.5:
                best_iou = iou_score
                best_idx = i
        if best_idx is None:
            continue

        idx = best_idx
        pose_raw = output[5:56, mask][:, idx]
        keypoints = pose_raw.reshape(17, 3)

        for i, (px, py, conf) in enumerate(keypoints):
            if conf < 0.3:
                continue
            px *= w0 / 640
            py *= h0 / 640
            cv2.circle(img, (int(px), int(py)), 3, (0, 0, 255), -1)
        for start_idx, end_idx in COCO_PAIRS:
            x_start, y_start, conf_start = keypoints[start_idx]
            x_end, y_end, conf_end = keypoints[end_idx]
            if conf_start > 0.3 and conf_end > 0.3:
                x_start = int(x_start * w0 / 640)
                y_start = int(y_start * h0 / 640)
                x_end = int(x_end * w0 / 640)
                y_end = int(y_end * h0 / 640)
                cv2.line(img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
    return img
#  Video setup
cap = cv2.VideoCapture(video_path)
# Get original video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_path = "output_pose.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'XVID' or 'avc1' if mp4v doesn't work
#fourcc = cv2.VideoWriter_fourcc(*"avc1") 
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h0, w0 = frame.shape[:2]

    #  Preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, input_size)
    img_input = img_resized.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0

    #  Inference

    intermediate_outputs = session.run(None, {input_name: img_input})

    # Print output shapes and types
    for i, output in enumerate(intermediate_outputs):
        print(f"Intermediate Output {i} shape: {output.shape}, dtype: {output.dtype}")

    #  Save outputs to pass to next subgraph
    Concat_output_0 = intermediate_outputs[0]
    Conv_output_0 = intermediate_outputs[1]
    sigmoid_output = intermediate_outputs[2]




    # Reshape inputs
    x1 = Concat_output_0.reshape(1, 17, 3, 8400)     # shape: 1×17×3×8400
    x2 = Conv_output_0.reshape(1, 4, 8400)           # shape: 1×4×8400

    # Slice operations
    # Placeholder slicing indices 
    slice1_concat_l = x1[:, :, :2, :]                         # shape: 1×17×2×8400
    slice2_concat_r = x1[:, :, 2:3, :]                         # shape: 1×17×3×8400 (no-op for completeness)
    slice3_conv_l = x2[:, :2, :]                            # shape: 1×2×8400
    slice4_conv_r = x2[:, 2:4, :]                            # shape: 1×2×8400

    # Arithmetic Operations
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
    sub1 = get_constant((3,640,640)) - slice3_conv_l
    print("slice4_conv_r shape",slice4_conv_r.shape,get_constant((3,640,640)).shape)
    print("slice3_conv_l shape",slice3_conv_l.shape,get_constant((3,640,640)).shape)
    # Right branch: arithmetic on x2 slices
    x8_conv_sub = add1 - sub1                     # shape: 1×2×8400
    x9_conv_add = sub1 + add1                             # shape: 1×2×8400


    x10_conv_r = x9_conv_add / 2     #add                                # shape: 1×2×8400



    # Concat -> shape: 1×4×8400
    x_concat2_conv = np.concatenate([x10_conv_r,x8_conv_sub], axis=1)
    #x11 = x_concat2 * np.ones((1, 4, 8400))          # Broadcasting Mul
    x11_conv = x_concat2_conv * get_mul_constant((3,640,640)) 
    print("x_concat2_conv shape",x_concat2_conv.shape,get_mul_constant((3,640,640)).shape)  
    # Final concat -> shape: 1×56×8400
    final_concat = np.concatenate([x11_conv,sigmoid_output,x_concat1_reshaped], axis=1)

    # Final output
    output = final_concat  # shape: 1×56×8400

    print("out shape",output.shape)



    output = output[0]

    boxes = output[0:4].T
    objectness = 1 / (1 + np.exp(-output[4]))
    class_scores = 1 / (1 + np.exp(-output[5:]))
    class_scores = class_scores.T

    confidences = objectness[:, None] * class_scores
    class_ids = np.argmax(confidences, axis=1)
    scores = np.max(confidences, axis=1)
    mask = scores > conf_threshold

    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    filtered_class_ids = class_ids[mask]


    # Scale and convert boxes
    filtered_boxes_scaled = filtered_boxes * np.array([w0 / 640, h0 / 640, w0 / 640, h0 / 640])
    filtered_boxes_scaled_xyxy = xywh_to_xyxy_np(filtered_boxes_scaled)

    dets = [
        [*filtered_boxes_scaled_xyxy[i], filtered_scores[i], filtered_class_ids[i]]
        for i in range(len(filtered_scores))
    ]

    # NMS

    final_detections = nms(dets)




    # Draw
    annotated = draw_detections(img.copy(), final_detections)
    pose_img = draw_pose(annotated, final_detections, output, mask)

    #out.write(pose_img)
    out.write(cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR))


    # --- Show
    cv2.imshow("Pose Estimation", cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
