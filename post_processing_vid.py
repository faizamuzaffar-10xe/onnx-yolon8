#For video boxes and pose full
import cv2
import numpy as np
import onnxruntime as ort

# --- Constants
video_path = "Trololo [iwGFalTRHDA].mp4"
onnx_path = "subgraph3.onnx"
input_size = (640, 640)
conf_threshold = 0.7
nms_threshold = 0.5

# --- Session
session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name

# --- Video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h0, w0 = frame.shape[:2]

    # --- Preprocess
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, input_size)
    img_input = img_resized.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255.0

    # --- Inference
    output = session.run(None, {input_name: img_input})[0]
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

    def xywh_to_xyxy_np(boxes):
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return np.stack([x1, y1, x2, y2], axis=1)

    # Scale and convert boxes
    filtered_boxes_scaled = filtered_boxes * np.array([w0 / 640, h0 / 640, w0 / 640, h0 / 640])
    filtered_boxes_scaled_xyxy = xywh_to_xyxy_np(filtered_boxes_scaled)

    dets = [
        [*filtered_boxes_scaled_xyxy[i], filtered_scores[i], filtered_class_ids[i]]
        for i in range(len(filtered_scores))
    ]

    # --- NMS
    def nms(dets, iou_thresh=0.5):
        if len(dets) == 0:
            return []
        boxes = np.array([d[:4] for d in dets])
        scores = np.array([d[4] for d in dets])
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=0.4,
            nms_threshold=iou_thresh
        )
        if len(indices) == 0:
            return []
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        return [dets[i] for i in indices]

    final_detections = nms(dets)

    # --- Draw detections
    def draw_detections(img, detections):
        for det in detections:
            x1, y1, x2, y2, score, class_id = map(int, det[:6])
            label = f"Class {class_id}: {score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img

    # --- Draw poses
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

    # --- Draw
    annotated = draw_detections(img.copy(), final_detections)
    pose_img = draw_pose(annotated, final_detections, output, mask)

    # --- Show
    cv2.imshow("Pose Estimation", cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
