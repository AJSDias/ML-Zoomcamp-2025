import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Load the ONNX model
session = ort.InferenceSession("hair_classifier_empty.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# URL of the image you want to test
url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"

# Download and preprocess
resp = requests.get(url)
img = Image.open(BytesIO(resp.content)).convert("RGB")
img = img.resize((200, 200), resample=Image.BILINEAR)

x = np.array(img, dtype=np.float32) / 255.0
x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
x = np.expand_dims(x, 0)        # Add batch dimension

print("Input shape:", x.shape)  # Should print: (1, 3, 200, 200)

# Run the model
y = session.run(None, {input_name: x})[0][0][0]

print("Prediction:", y)
