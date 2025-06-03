import numpy as np

import cv2 
import onnxruntime as ort
import numpy as np
import numpy
# Dummy input tensors
# Concat_output_0 = np.random.rand(1, 51, 8400)
# Conv_output_0 = np.random.rand(1, 1, 4, 8400)
# sigmoid_output_0 = np.random.rand(1,1,8400)


# 🔧 STEP 1: Preprocess the input image

def preprocess_image(img_path, input_size=(640, 640)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img.transpose(2, 0, 1)  # HWC → CHW
    img = np.expand_dims(img, axis=0)  # CHW → BCHW
    img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
    #print(img.shape, img.dtype)

    return img


# Load preprocessed image
img_input = preprocess_image("bus.jpg")  # Replace with your image path

# 🔧 STEP 2: Load first subgraph and run inference
# session = ort.InferenceSession("yolov8n_first_subgraph.onnx")
try:
    session = ort.InferenceSession("subgraph3.onnx")
    #print("Session created successfully.")
except Exception as e:
    print("Failed to create session:", e)

# # Get input name for the session (usually 'images' or similar)
input_name = session.get_inputs()[0].name
print(f"Input name: {input_name}")

# Run inference on the preprocessed image
output0_subgraph3 = session.run(None, {input_name: img_input})

# 🔍 STEP 3: Print output shapes and types
# 🔍 STEP 3: Print output shapes and types
for i, output in enumerate(output0_subgraph3):
    print(f"output0_subgraph3 Output {i} shape: {output.shape}, dtype: {output.dtype}")




output = output[0]             # remove batch dim → (56, 8400)

boxes = output[0:4].T                    # (8400, 4) → cx, cy, w, h
objectness = 1 / (1 + np.exp(-output[4]))  # sigmoid (8400,)
class_scores = 1 / (1 + np.exp(-output[5:]))  # sigmoid (51, 8400)
class_scores = class_scores.T  # (8400, 51)


input_size = (640, 640)
original_img = cv2.imread("bus.jpg")
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
h0, w0 = original_img.shape[:2]


confidences = objectness[:, None] * class_scores  # shape: (8400, 51)
class_ids = np.argmax(confidences, axis=1)        # shape: (8400,)
scores = np.max(confidences, axis=1)              # shape: (8400,)

# Filter out low confidence boxes
conf_threshold = 0.7  # INCREASED from 0.3
mask = scores > conf_threshold
filtered_boxes = boxes[mask]         # (N, 4)
filtered_scores = scores[mask]       # (N,)
filtered_class_ids = class_ids[mask] # (N,)

# Convert boxes from cx,cy,w,h to x1,y1,x2,y2
def xywh_to_xyxy(xywh):
    cx, cy, w, h = xywh
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

xyxy_boxes = np.array([xywh_to_xyxy(b) for b in filtered_boxes])

# Scale boxes to original image size
scale = np.array([w0 / 640, h0 / 640, w0 / 640, h0 / 640])
xyxy_boxes = xyxy_boxes * scale

# Format detections for NMS
dets = []
for i in range(len(xyxy_boxes)):
    dets.append([
        *xyxy_boxes[i], filtered_scores[i], filtered_class_ids[i]
    ])

# NMS using OpenCV
# def nms(dets, iou_thresh=0.5):
#     boxes = [d[:4] for d in dets]
#     scores = [d[4] for d in dets]
#     indices = cv2.dnn.NMSBoxes(boxes, scores, 0.3, iou_thresh)
#     return [dets[i[0]] if isinstance(i, (list, np.ndarray)) else dets[i] for i in indices]
def nms(dets, iou_thresh=0.5):
    if len(dets) == 0:
        return []

    boxes = np.array([d[:4] for d in dets])  # x1, y1, x2, y2
    scores = np.array([d[4] for d in dets])
    
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes.tolist(), scores=scores.tolist(),
        score_threshold=0.4, nms_threshold=iou_thresh
    )
    
    if len(indices) == 0:
        return []
    
    # Flatten & return filtered detections
    if isinstance(indices, np.ndarray):
        indices = indices.flatten()
    return [dets[i] for i in indices]


final_detections = nms(dets)
def draw_detections(img, detections):
    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"Class {int(class_id)}: {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img

result_img = draw_detections(original_img.copy(), final_detections)
cv2.imwrite("output_subgraph3.jpg", cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
print("Saved image with detections to output_subgraph3.jpg")

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

def xywh_to_xyxy_np(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)

def check_num_keypoints(keypoints):
    # keypoints is expected shape: (N_people, 17, 3)
    for i, person_kps in enumerate(keypoints):
        num_kps = person_kps.shape[0]
        if num_kps != 17:
            print(f"Person {i} has {num_kps} keypoints instead of 17!")
        else:
            print(f"Person {i} has 17 keypoints ✅")
def check_keypoints_bounds(keypoints, img_width, img_height):
    for i, person_kps in enumerate(keypoints):
        for j, (x, y, conf) in enumerate(person_kps):
            if not (0 <= x <= img_width) or not (0 <= y <= img_height):
                print(f"Person {i}, keypoint {j} out of bounds: x={x}, y={y}")

def check_confidence_scores(keypoints):
    for i, person_kps in enumerate(keypoints):
        for j, (x, y, conf) in enumerate(person_kps):
            if conf < 0 or conf > 1:
                print(f"Person {i}, keypoint {j} confidence out of range: {conf}")


COCO_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Right arm
    (0, 5), (5, 6), (6, 7), (7, 8),       # Left arm
    (0, 9), (9, 10), (10, 11),             # Right leg
    (0, 12), (12, 13), (13, 14),           # Left leg
    (0, 15), (15, 16)                      # Neck/Head connections
]
# def draw_pose(img, dets, output, mask, input_shape=(640, 640), original_shape=(640, 640)):
#     print("xxxxx")
#     for det in dets:
        
#         x1, y1, x2, y2, score, class_id = det
#         cx = (x1 + x2) / 2
#         cy = (y1 + y2) / 2

#         # Find matching index from filtered_boxes
#         # idx = None
#         # for i, box in enumerate(filtered_boxes):
#         #     bx, by, _, _ = box
            
#         #     if np.isclose(bx, cx, atol=1) and np.isclose(by, cy, atol=1):
#         #         idx = i
#         #         break

#         # if idx is None:
#         #     print("xxxxx none")
#         #     continue


#         # Inside draw_pose
#         best_iou = 0
#         best_idx = None
#         for i, box in enumerate(filtered_boxes_scaled):

#         #for i, box in enumerate(filtered_boxes):
#             iou_score = iou(det[:4], box)
#             if iou_score > best_iou and iou_score > 0.5:
#                 best_iou = iou_score
#                 best_idx = i

#         if best_idx is None:
#             print("No match found for this detection.")
#             continue

#         idx = best_idx
#         # Get keypoints from output (shape: (51, 8400))
#         pose_raw = output[5:56, mask][:, idx]  # shape: (51,)
#         keypoints = pose_raw.reshape(17, 3)
#         # print("Pose keypoints (raw):")
#         # print(keypoints)
        


#         for i, (px, py, conf) in enumerate(keypoints):
#             if conf < 0.3:
#                 continue
#             px *= original_shape[1] / input_shape[0]
#             py *= original_shape[0] / input_shape[1]
#             cv2.circle(img, (int(px), int(py)), 3, (0, 0, 255), -1)
#             cv2.putText(img, str(i), (int(px)+2, int(py)-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
#     return img

def draw_pose(img, dets, output, mask, input_shape=(640, 640), original_shape=(640, 640)):
    print("xxxxx")

    # COCO skeleton pairs (keypoint indices)
    COCO_PAIRS = [
        (5, 7), (7, 9),        # Left arm
        (6, 8), (8, 10),       # Right arm
        (11, 13), (13, 15),    # Left leg
        (12, 14), (14, 16),    # Right leg
        (5, 6),                # Shoulders
        (11, 12),              # Hips
        (5, 11), (6, 12),      # Torso sides
        (0, 1), (1, 3),        # Left face
        (0, 2), (2, 4),        # Right face
        (0, 5), (0, 6)         # Nose to shoulders
    ]


    for det in dets:
        x1, y1, x2, y2, score, class_id = det
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        best_iou = 0
        best_idx = None
        for i, box in enumerate(filtered_boxes_scaled):
            iou_score = iou(det[:4], box)
            if iou_score > best_iou and iou_score > 0.5:
                best_iou = iou_score
                best_idx = i

        if best_idx is None:
            print("No match found for this detection.")
            continue

        idx = best_idx
        pose_raw = output[5:56, mask][:, idx]  # (51,)
        keypoints = pose_raw.reshape(17, 3)

        # Scale keypoints to original image size
        for i, (px, py, conf) in enumerate(keypoints):
            if conf < 0.3:
                continue
            px *= original_shape[1] / input_shape[0]
            py *= original_shape[0] / input_shape[1]
            cv2.circle(img, (int(px), int(py)), 3, (0, 0, 255), -1)
            cv2.putText(img, str(i), (int(px)+2, int(py)-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        # Draw skeleton lines
        for start_idx, end_idx in COCO_PAIRS:
            x_start, y_start, conf_start = keypoints[start_idx]
            x_end, y_end, conf_end = keypoints[end_idx]

            if conf_start > 0.3 and conf_end > 0.3:
                x_start = int(x_start * original_shape[1] / input_shape[0])
                y_start = int(y_start * original_shape[0] / input_shape[1])
                x_end = int(x_end * original_shape[1] / input_shape[0])
                y_end = int(y_end * original_shape[0] / input_shape[1])

                cv2.line(img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)  # Blue line

    return img

# Scale filtered_boxes (cx, cy, w, h) to original image size
filtered_boxes_scaled = filtered_boxes * np.array([w0 / 640, h0 / 640, w0 / 640, h0 / 640])

filtered_boxes_scaled_xyxy = xywh_to_xyxy_np(filtered_boxes_scaled)
filtered_boxes_scaled=filtered_boxes_scaled_xyxy
result_img_with_pose = draw_pose(
    result_img.copy(),
    final_detections,
    output,
    mask,
    input_shape=(640, 640),
    original_shape=(h0, w0)
)





cv2.imwrite("output_pose.jpg", cv2.cvtColor(result_img_with_pose, cv2.COLOR_RGB2BGR))
print("Saved image with pose keypoints to output_pose.jpg")

