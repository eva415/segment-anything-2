import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# Start a mixed-precision context for faster inference (if supported)
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# Enable TensorFloat32 if on Ampere or newer GPUs (for faster matrix ops)
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Load the model and its configuration
sam2_checkpoint = "/home/imml/git/research_2025/sam2/segment-anything-2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

# Visualize a binary mask on a frame
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Visualize positive/negative click points
def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolors='white')
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolors='white')

# Directory containing frames from a video
video_dir = "/home/imml/git/research_2025/sam2/segment-anything-2/video"

# Read all image frame filenames
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))  # Sort numerically

# Show first frame for selecting an object
frame_idx = 0
plt.figure(figsize=(12,8))
plt.title(f"frame{frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))
plt.show()

input("Press Enter to continue...")

# Initialize SAM2 model for inference
inference_state = predictor.init_state(video_path=video_dir)
predictor.reset_state(inference_state)

# Define the frame and object ID for annotation
ann_frame_idx = 0
ann_obj_id = 1

# Coordinates for the object to track (user should adjust)
points = np.array([[478,169], [533, 182]], dtype=np.float32)
labels = np.array([1,1], np.int32)

# Add these points to initialize object tracking
_, out_obj_ids, out_mask_logits = predictor.add_new_points(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# Show annotated object
plt.figure(figsize=(12,8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
plt.show()

input("Press Enter to continue...")

video_segments = {}

# Minimum mask area to consider the object still visible
MIN_MASK_AREA = 50

# Track object IDs that have been removed from tracking
disabled_obj_ids = set()

# Track objects through the video
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    frame_results = {}

    for i, out_obj_id in enumerate(out_obj_ids):
        # Skip object if we've already decided to stop tracking it
        if out_obj_id in disabled_obj_ids:
            continue

        # Convert logits to binary mask
        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
        mask_area = np.sum(mask)

        # If the mask is too small, stop tracking this object
        if mask_area < MIN_MASK_AREA:
            print(f"TOO SMALL, NO MORE PEAR :( (frame {out_frame_idx})")
            disabled_obj_ids.add(out_obj_id)
            continue

        frame_results[out_obj_id] = mask

    video_segments[out_frame_idx] = frame_results


# Visualization settings
vis_frame_stride = 1
plt.close("all")

fig = plt.figure(figsize=(6, 4))
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.title(f"frame {out_frame_idx}")
    im = plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])), animated=True)
    
    # Show only active (non-disappeared) objects
    for out_obj_id, out_mask in video_segments.get(out_frame_idx, {}).items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

    plt.savefig(f'output/s{out_frame_idx}.png')
