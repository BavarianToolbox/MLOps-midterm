# FROM tensorflow/tensorflow:latest-gpu
FROM tensorflow/tensorflow:nightly-gpu

ADD train.py reqs_train.txt ./

RUN pip install --upgrade pip
RUN pip install -r reqs_train.txt

ENTRYPOINT ["python", "train.py"]

#CMD ["python", "train.py", "-p", "$PROJ", "-c", "$CFG"]

#CMD python train.py -p $PROJ -c $CFG

# CMD python train.py -p midterm-train -c cfg_00001.json
