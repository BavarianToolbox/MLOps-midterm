# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
# FROM tensorflow/tensorflow:latest
# FROM python:3.7
FROM tensorflow/tensorflow:nightly

# python:3.7 version
# RUN pip install fastapi uvicorn tensorflow pandas Pillow python-multipart wandb google-cloud-storage matplotlib tf-nightly
# tf-nightly version
RUN pip install Pillow pandas fastapi uvicorn python-multipart wandb google-cloud-storage matplotlib

EXPOSE 80

COPY ./app /app

#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT
