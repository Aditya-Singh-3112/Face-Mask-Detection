import numpy as np
import gradio as gr
import tensorflow
from tensorflow.keras.models import model_from_json

with open('model_json.json', 'r') as f:
    model_json = f.read()

model = model_from_json(model_json)

model.load_weights('face_mask_detection.h5')

def model_fn(img):
    img_resized = np.resize(img, (155, 155, 3))
    img = np.expand_dims(img_resized, axis=0)
    img = img/255.0
    y = model.predict(img)
    out = y[0][0]*1000000

    if out < 0.001:
        return "Mask Detected"
    else:
        return "Mask Not Detected"

ui = gr.Interface(fn = model_fn, inputs = [gr.Image(image_mode='RGB')], outputs = ['text'])

ui.launch()