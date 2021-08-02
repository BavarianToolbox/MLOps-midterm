FROM tensorflow/tensorflow:latest-gpu

ADD train.py train_reqs.txt ./

RUN pip install --use-feature=2020-resolver -r train_reqs.txt

ENTRYPOINT ["python", "./train.py"]
