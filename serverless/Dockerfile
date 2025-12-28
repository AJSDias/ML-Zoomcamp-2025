FROM agrigorev/model-2025-hairstyle:v1

RUN pip install --no-cache-dir \
    onnxruntime \
    pillow \
    numpy \
    requests

COPY lambda_function.py .

CMD ["lambda_function.handler"]