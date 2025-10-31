FROM agrigorev/zoomcamp-model:2025

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml .python-version uv.lock /app/
RUN uv sync --locked

#COPY deployment/homework/predict_app.py deployment/homework/pipeline_v1.bin /app/
COPY deployment/homework/predict_app.py /app/


# Install uvicorn explicitly
RUN uv pip install uvicorn

EXPOSE 9696

ENTRYPOINT ["uv", "run", "uvicorn", "predict_app:app", "--host", "0.0.0.0", "--port", "9696"]
