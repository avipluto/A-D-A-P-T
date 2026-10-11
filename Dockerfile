# Public ADAPT server without PyTorch: NumPy inference with the weights in models/.
# Build:  docker build -t adapt .        Run:  docker run -p 8000:8000 -e PORT=8000 adapt
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/home/user/app/src \
    ADAPT_RUNTIME=numpy \
    ADAPT_PUBLIC=1 \
    ADAPT_DATA_DIR=/tmp/adapt-data \
    ADAPT_FEEDBACK_DB=/tmp/adapt-data/feedback/feedback.sqlite
WORKDIR /home/user/app

COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir --user -r requirements-deploy.txt

COPY --chown=user src ./src
COPY --chown=user models ./models

# Render (and most hosts) pass the port in $PORT
CMD ["sh", "-c", "python -m uvicorn --factory adapt.api.app:create_app --host 0.0.0.0 --port ${PORT:-10000} --proxy-headers --forwarded-allow-ips '*'"]
