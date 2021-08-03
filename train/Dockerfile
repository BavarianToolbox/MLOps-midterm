FROM tensorflow/tensorflow:latest-gpu

ADD train.py reqs_train.txt ./

RUN pip install --upgrade pip
RUN pip install -r reqs_train.txt

#ENTRYPOINT ["python", "train.py"]

#CMD ["python", "train.py", "-p", "$PROJ", "-c", "$CFG"]

#CMD python train.py -p $PROJ -c $CFG

CMD python train.py -p midterm-train-test -c cfg_00001.json
