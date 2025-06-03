# from PIL import Image

# from ultralytics import YOLO

# # Load a pretrained YOLO11n model
# model = YOLO("yolov8n-pose.onnx")

# # Run inference on 'bus.jpg'
# results = model(["https://ultralytics.com/images/bus.jpg", "https://ultralytics.com/images/zidane.jpg"])  # results list

# # Visualize the results
# for i, r in enumerate(results):
#     # Plot results image
#     im_bgr = r.plot()  # BGR-order numpy array
#     im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

#     # Show results to screen (in supported environments)
#     r.show()

#     # Save results to disk
#     r.save(filename=f"results{i}.jpg")

from ultralytics import YOLO
import cv2
# Load a model
model = YOLO("yolov8n-pose.pt")  # load an official model


# Predict with the model
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Access the results

for i, result in enumerate(results):
    # Get keypoint data if needed
    xy = result.keypoints.xy  # x and y coordinates
    xyn = result.keypoints.xyn  # normalized
    kpts = result.keypoints.data  # x, y, visibility (if available)

    # Get rendered image with keypoints
    rendered_img = result.plot()

    # Save image
    cv2.imwrite(f"pose_result_{i}.jpg", rendered_img)
