import streamlit as st
from PIL import Image
import torch
import numpy as np
import os
import sys


from yolov5.utils.general import non_max_suppression

from yolov5.utils.augmentations import letterbox

from yolov5.models.experimental import attempt_load
sys.path.insert(0, os.path.abspath('./yolov5'))


device = torch.device('cpu')

# load yolo model
model = attempt_load('./yolov5/yolov5s.pt', device=device)
model.eval()


def detect_objects(image):
    # preprocess the image
    img = letterbox(image, new_shape=640)[0]
    img = img.transpose((2, 0, 1))  # Convert to CHW format
    img = np.ascontiguousarray(img)

    # convert to tensor
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)


    pred = model(img)[0]

    # apply non-maximum suppression
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

    # extract names of detected objects
    names = model.module.names if hasattr(model, 'module') else model.names
    detected_components = []

    for det in pred:
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                label = names[int(cls)]
                detected_components.append(label)

    return detected_components

## upload image & display it
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
## analyse image
    if st.button('Analyse Image'):
        components = detect_objects(np.array(image))
        st.write("Detected Components:")
        for component in components:
            st.write(f"- {component}")