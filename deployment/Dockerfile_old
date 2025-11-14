FROM python:3.12.4-slim-bookworm

# Install uv
#RUN pip install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set work directory
WORKDIR /app

# Copy project metadata and install deps
COPY pyproject.toml .python-version uv.lock /app/
RUN uv sync --locked

# Copy app files
COPY deployment/workshop/predict.py deployment/workshop/model.bin /app/

# Expose port
EXPOSE 9696

# Run with uvicorn
ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]



