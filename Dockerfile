# FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime
FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

COPY app/ ./app/
COPY models/ ./models/
COPY requirements.txt .
COPY README.md .

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 8000 8501

CMD uvicorn app.appfast:app --host 0.0.0.0 --port 8000 & streamlit run app/new_streamlit_app.py --server.port 8501 --server.address 0.0.0.0