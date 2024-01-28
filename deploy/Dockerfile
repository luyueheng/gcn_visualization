FROM ubuntu:20.04

WORKDIR /root

RUN apt update

RUN apt install -y python3 curl

RUN apt install -y python3.8-distutils ffmpeg libsm6 libxext6

RUN curl https://bootstrap.pypa.io/get-pip.py > get-pip.py

RUN python3 get-pip.py

COPY requirements.txt /root/

RUN pip install -r requirements.txt

COPY model /root/model
COPY visualization_flask /root/visualization_flask

WORKDIR /root/visualization_flask

ENV FLASK_APP=/root/visualization_flask/gcn_server.py

CMD ["flask", "run", "-h", "0.0.0.0", "-p", "80"]
