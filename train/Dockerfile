FROM tensorflow/tensorflow:latest-gpu

ADD train.py reqs_train.txt ./

RUN pip install --use-feature=2020-resolver -r reqs_train.txt

ENTRYPOINT ["python", "./train.py"]
