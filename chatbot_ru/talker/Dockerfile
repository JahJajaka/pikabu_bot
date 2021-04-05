FROM python:3.8.6-slim-buster

COPY requirements.txt /tmp/

RUN apt-get update \
    && apt-get install -y postgresql-server-dev-11 gcc python3-dev musl-dev \
    && apt-get install -y gettext-base \
    && pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt 
RUN apt-get install -y bash \
                   build-essential \
                   git \
                   ca-certificates && \
    rm -rf /var/lib/apt/lists
RUN python3 -m pip install jupyter torch transformers[torch]

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir /talker
COPY ./ /talker/

WORKDIR /talker
CMD ["python3", "bot.py"]