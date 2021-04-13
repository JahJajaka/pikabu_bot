FROM python:3.8.6-slim-buster

COPY requirements.txt /tmp/

RUN apt-get update \
    && apt-get install -y gettext-base \
    && pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt 

RUN mkdir /telegram
COPY ./ /telegram/

WORKDIR /telegram

CMD ["python3", "bot.py"]