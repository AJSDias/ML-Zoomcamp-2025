import onnxruntime as ort
import numpy as np
from PIL import Image
import requests
from io import BytesIO


session = ort.InferenceSession(
    "hair_classifier_empty.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

def handler(event, context):
    url = event["url"]

    print ('HELLLLLLLLOOOOOOOOOOO')
    # Download and convert image
    resp = requests.get(url)
    img = Image.open(BytesIO(resp.content)).convert("RGB")

    # Preprocess
    img = img.resize((200, 200))
    x = np.array(img).astype("float32") / 255.0
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    x = np.expand_dims(x, axis=0)   # add batch dimension

    print("Input shape:", x.shape)  # <-- debug

    # Run model
    y = session.run(None, {input_name: x})[0][0][0]

    return float(y)
