import os
import cv2
import torch
import numpy as np

from DexiNed.model import DexiNed
from utils.utilities import *

CHECKPOINT = 'DexiNed/checkpoints/BIPED/10/10_model.pth'

def model(
        image_path:str,
):

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MEAN_BGR = [103.939, 116.779, 123.68]

    model = DexiNed().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        image, h, w = resize4patch(image, h, w)

        # Preprocessing
        image = image.astype(np.float32)
        image -= np.array(MEAN_BGR)
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).unsqueeze(0).to(DEVICE)

        preds = model(image)

        # Average output
        edges = [torch.sigmoid(p) for p in preds]
        avg_edge = torch.mean(torch.stack(edges), dim=0)
        edge = avg_edge.squeeze().cpu().numpy()
        edge = (255 * edge).astype(np.uint8)

        base, ext = os.path.splitext(image_path)
        save_path = base + '_edge' + ext
        cv2.imwrite(save_path, edge)

        return edge

IMG_PATH = 'demo/Avatar.jpg'
model(IMG_PATH)