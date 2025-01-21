import cv2
import torch
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os

print("OpenCV version:", cv2.__version__)
print(torch.cuda.is_available())


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH=r"D:\Melbin\G5_ML_model\SAM_model\Models\sam_vit_h_4b8939.pth"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)

IMAGE_NAME = r"input_files/page_1.png"
IMAGE_PATH = os.path.join( "data", IMAGE_NAME)

import cv2
import supervision as sv

image_bgr = cv2.imread(IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

sam_result = mask_generator.generate(image_rgb)

print(sam_result[0].keys())