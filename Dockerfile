FROM tensorflow/tensorflow:latest-gpu

ADD train.py train_reqs.txt ./

RUN pip install -r train_reqs.txt

ENTRYPOINT ["python", "./train.py"]
