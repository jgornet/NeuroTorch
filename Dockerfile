FROM python:3.6.6-alpine

WORKDIR /app

ADD . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt

EXPOSE 6006

ENV NAME NeuroTorch
