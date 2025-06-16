

#For video boxes and pose full
import cv2
import numpy as np
import numpy
import onnxruntime as ort






def preprocess_image(frame, input_size=640):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if h > w:
        scale = input_size / h
        new_h = input_size
        new_w = int(w * scale)
    else:
        scale = input_size / w
        new_w = input_size
        new_h = int(h * scale)

    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    padded_img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    pad_top = (input_size - new_h) // 2
    pad_left = (input_size - new_w) // 2
    padded_img[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized_img

    img_tensor = padded_img.transpose(2, 0, 1)
    img_tensor = np.expand_dims(img_tensor, axis=0).astype(np.float32) / 255.0

    return img_tensor, scale, pad_left, pad_top


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

def draw_detections(img, detections,class_scores):
    for det in detections:
        x1, y1, x2, y2, score, class_id = det[:6]
        if int(class_id) != 0:
            continue  # Only show person (class ID 0)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"Person: {score * 100:.1f}%"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Person : {str(class_scores[0][0])[:8]}", (x1, y1 - 5),
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

def get_pose_indices(dets, filtered_boxes_scaled):
    matched_indices = []
    for det in dets:
        best_iou = 0
        best_idx = None
        for i, box in enumerate(filtered_boxes_scaled):
            iou_score = iou(det[:4], xywh_to_xyxy_np(np.expand_dims(box, axis=0))[0])
            if iou_score > best_iou and iou_score > 0.5:
                best_iou = iou_score
                best_idx = i
        matched_indices.append(best_idx)
    return matched_indices

def draw_pose(img, dets, output, mask, scale, pad_left, pad_top, matched_indices):
    COCO_PAIRS = [
        (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (5, 6), (11, 12), (5, 11), (6, 12),
        (0, 1), (1, 3), (0, 2), (2, 4),
        (0, 5), (0, 6)
    ]
    for det, idx in zip(dets, matched_indices):
        if idx is None:
            continue
        pose_raw = output[5:56, mask][:, idx]
        keypoints = pose_raw.reshape(17, 3)

        for i, (px, py, conf) in enumerate(keypoints):
            if conf < 0.3:
                continue
            px = (px - pad_left) / scale
            py = (py - pad_top) / scale
            cv2.circle(img, (int(px), int(py)), 3, (0, 0, 255), -1)

        for start_idx, end_idx in COCO_PAIRS:
            x_start, y_start, conf_start = keypoints[start_idx]
            x_end, y_end, conf_end = keypoints[end_idx]
            if conf_start > 0.3 and conf_end > 0.3:
                x_start = (x_start - pad_left) / scale
                y_start = (y_start - pad_top) / scale
                x_end = (x_end - pad_left) / scale
                y_end = (y_end - pad_top) / scale
                cv2.line(img, (int(x_start), int(y_start)), (int(x_end), int(y_end)), (255, 0, 0), 2)
    return img


def main(video_path = "Trololo [iwGFalTRHDA].mp4",
    onnx_path = "subgraph1.onnx",
    input_size = (640, 640),
    conf_threshold = 0.6,
    nms_threshold = 0.5):
    #  Video setup
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_path = "output_pose.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h0, w0 = frame.shape[:2]
        #Preprocess output
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #img_input = preprocess_image(frame,(640, 640))
        #img_input,new_w,new_h = preprocess_image(frame, 640)
        img_input, scale, pad_left, pad_top = preprocess_image(frame)


        intermediate_outputs = session.run(None, {input_name: img_input})

        Concat_output_0 = intermediate_outputs[0]
        Conv_output_0 = intermediate_outputs[1]
        sigmoid_output = intermediate_outputs[2]

        x1 = Concat_output_0.reshape(1, 17, 3, 8400)
        x2 = Conv_output_0.reshape(1, 4, 8400)

        slice1_concat_l = x1[:, :, :2, :]
        slice2_concat_r = x1[:, :, 2:3, :]
        slice3_conv_l = x2[:, :2, :]
        slice4_conv_r = x2[:, 2:4, :]

        x3_concat_l = slice1_concat_l * 2
        x4_concat_l = x3_concat_l + get_constant_pose()
        x5_concat_l = x4_concat_l * get_mul_constant((3,640,640))
        x6_concat_r = slice2_concat_r
        x7_concat_r = 1 / (1 + np.exp(-x6_concat_r))
        x_concat1 = np.concatenate([x5_concat_l, x7_concat_r], axis=2)
        x_concat1_reshaped = x_concat1.reshape(1, 51, 8400)

        add1 = slice4_conv_r + get_constant((3,640,640))
        sub1 = get_constant((3,640,640)) - slice3_conv_l
        x8_conv_sub = add1 - sub1
        x9_conv_add = sub1 + add1
        x10_conv_r = x9_conv_add / 2
        x_concat2_conv = np.concatenate([x10_conv_r,x8_conv_sub], axis=1)
        x11_conv = x_concat2_conv * get_mul_constant((3,640,640))
        final_concat = np.concatenate([x11_conv,sigmoid_output,x_concat1_reshaped], axis=1)

        output = final_concat[0]

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
        # Adjust for scale and padding (xywh format)
        filtered_boxes_scaled = filtered_boxes.copy()
        filtered_boxes_scaled[:, 0] = (filtered_boxes_scaled[:, 0] - pad_left) / scale  # cx
        filtered_boxes_scaled[:, 1] = (filtered_boxes_scaled[:, 1] - pad_top) / scale   # cy
        filtered_boxes_scaled[:, 2] = filtered_boxes_scaled[:, 2] / scale               # w
        filtered_boxes_scaled[:, 3] = filtered_boxes_scaled[:, 3] / scale               # h

        #filtered_boxes_scaled = filtered_boxes * np.array([w0 / 640, h0 / 640, w0 / 640, h0 / 640])
        filtered_boxes_scaled_xyxy = xywh_to_xyxy_np(filtered_boxes_scaled)

        dets = [
            [*filtered_boxes_scaled_xyxy[i], filtered_scores[i], filtered_class_ids[i]]
            for i in range(len(filtered_scores))
        ]

        final_detections = nms(dets)
        annotated = draw_detections(img.copy(), final_detections,class_scores)
        matched_indices = get_pose_indices(final_detections, filtered_boxes_scaled)
        pose_img = draw_pose(annotated, final_detections, output, mask, scale, pad_left, pad_top, matched_indices)

        out.write(cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR))
        cv2.imshow("Pose Estimation", cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
