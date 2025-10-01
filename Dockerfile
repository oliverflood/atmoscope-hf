FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=7860

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY app.py ./app.py
COPY clouds ./clouds
RUN mkdir -p models
COPY models/clouds_bundle.pt ./models/clouds_bundle.pt

CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0
