import os
import json
import cv2
import torch
import numpy as np

from DexiNed.model import DexiNed
from utils.utilities import *

JSON_PATH = 'data/json/imd2020_index.json'
DATASET_ROOT = 'IMD2020'
CHECKPOINT = 'DexiNed/checkpoints/BIPED/10/10_model.pth'
OUTPUT_DIR = 'IMD2020_edges'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MEAN_BGR = [103.939, 116.779, 123.68]

model = DexiNed().to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

image_paths = []
mask_paths = []

for image_folder, sample in data.items():
    if 'original' in sample and 'manipulated' in sample:
        image_paths.append(sample['original'])

        for manip in sample['manipulated']:
            it = iter(manip)
            image_paths.append(manip[next(it)])
            mask_paths.append(manip[next(it)])

# Testing
with torch.no_grad():
    for img_path in image_paths:

        image = cv2.imread(img_path)
        h, w = image.shape[:2]
        image, h, w = resize4patch(image, h, w)

        # Preprocessing
        image = image.astype(np.float32)
        image -= np.array(MEAN_BGR) # Normalization
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE) # Into batch

        preds = model(image)

        # Average output
        edges = [torch.sigmoid(p) for p in preds]
        avg_edge = torch.mean(torch.stack(edges), dim=0)
        edge = avg_edge.squeeze().cpu().numpy()
        edge = (255 * edge).astype(np.uint8)

        relative_path = os.path.relpath(img_path, DATASET_ROOT)
        relative_path = os.path.splitext(relative_path)[0] + '.png'

        save_path = os.path.join(OUTPUT_DIR, relative_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        cv2.imwrite(save_path, edge)

# Copy masks to new dataset
for mask_path in mask_paths:
    
    full_path = mask_path
    if not os.path.isabs(full_path):
        full_path = os.path.join(DATASET_ROOT, mask_path)

    image = cv2.imread(full_path)

    relative_path = os.path.relpath(full_path, DATASET_ROOT)
    relative_path = os.path.splitext(relative_path)[0] + '.png'

    save_path = os.path.join(OUTPUT_DIR, relative_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cv2.imwrite(save_path, image)

print('Successfully Done!')