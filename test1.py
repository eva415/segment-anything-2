import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

from ultralytics import YOLO
import cv2

# — make sure output dir exists —
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# -- YOLO SETUP --
yolo_model = YOLO("/home/imml/git/research_2025/ultralytics/runs/detect/train3/weights/best.pt")

def get_centroids_from_image(image, draw=False):
    results = yolo_model(image, conf=0.1)
    centroids = []
    image_draw = image.copy() if draw else None

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centroids.append((cx, cy))

            if draw:
                cv2.rectangle(image_draw, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.circle(image_draw, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(image_draw, f"({cx},{cy})", (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if draw:
        return centroids, image_draw
    else:
        return centroids

# -- SAM 2 SETUP --
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

sam2_checkpoint = "/home/imml/git/research_2025/sam2/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# Precompute a fixed color per object ID using matplotlib's tab20 colormap
import matplotlib.cm as cm

NUM_COLORS = 20
color_map = cm.get_cmap('tab20b', NUM_COLORS)
obj_id_to_color = {}

def get_color_for_obj(obj_id):
    if obj_id not in obj_id_to_color:
        color = color_map(obj_id % NUM_COLORS)  # RGBA
        obj_id_to_color[obj_id] = np.array([*color[:3], 0.6])  # Drop alpha, add fixed transparency
    return obj_id_to_color[obj_id]

def show_mask(mask, ax, obj_id=None):
    color = get_color_for_obj(obj_id if obj_id is not None else 0)
    h, w = mask.shape[-2:]
    ax.imshow(mask.reshape(h, w, 1) * color.reshape(1,1,-1))

def show_points(coords, labels, ax, marker_size=200):
    pos = coords[labels==1]
    neg = coords[labels==0]
    ax.scatter(pos[:,0], pos[:,1], marker='*', s=marker_size, edgecolors='white', color='green')
    ax.scatter(neg[:,0], neg[:,1], marker='*', s=marker_size, edgecolors='white', color='red')

# -- MAIN LOOP --

video_dir = "/home/imml/git/research_2025/sam2/segment-anything-2/video"
frame_names = [p for p in os.listdir(video_dir)
               if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# First frame: detect pears
first_frame_path = os.path.join(video_dir, frame_names[0])
first_frame = cv2.imread(first_frame_path)
centroids, img_with_detections = get_centroids_from_image(first_frame, draw=True)
print(f"Detected {len(centroids)} pears at {centroids}")

# Save YOLO detection overlay
cv2.imwrite(os.path.join(output_dir, "frame0_detections.jpg"), img_with_detections)
print(f"Saved YOLO detections to {os.path.join(output_dir, 'frame0_detections.jpg')}")

# (Optional) preview
cv2.imshow("YOLO detections", img_with_detections)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# Prepare SAM2 inputs
points = np.array(centroids, dtype=np.float32)
labels = np.ones(len(points), dtype=np.int32)

inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

for i, pt in enumerate(points):
    _, _, _ = predictor.add_new_points(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=i+1,
        points=pt.reshape(1,2),
        labels=np.array([1], dtype=np.int32),
    )

MIN_MASK_AREA = 50
disabled_obj_ids = set()
video_segments = {}

for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
    results = {}
    for i, obj_id in enumerate(obj_ids):
        if obj_id in disabled_obj_ids:
            continue
        mask = (mask_logits[i] > 0.0).cpu().numpy()
        if mask.sum() < MIN_MASK_AREA:
            disabled_obj_ids.add(obj_id)
            print(f"Stopping tracking of pear #{obj_id} at frame {frame_idx}")
            continue
        results[obj_id] = mask
    video_segments[frame_idx] = results

# Visualize & save each overlaid frame
plt.close('all')
fig = plt.figure(figsize=(8,6))

for idx in range(len(frame_names)):
    plt.clf()
    plt.title(f"frame {idx}")
    img = Image.open(os.path.join(video_dir, frame_names[idx]))
    plt.imshow(img)
    for obj_id, mask in video_segments.get(idx, {}).items():
        show_mask(mask, plt.gca(), obj_id=obj_id)
    save_path = os.path.join(output_dir, f"frame_{idx:03d}.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved overlay to {save_path}")
    plt.pause(0.05)

plt.show()
