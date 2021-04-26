FROM python:3.8.6-slim-buster

COPY requirements.txt /tmp/

RUN apt-get update \
    && apt-get install -y postgresql-server-dev-11 gcc python3-dev musl-dev \
    && apt-get install -y gettext-base \
    && apt-get install -y bash \
    && rm -rf /var/lib/apt/lists
RUN pip install --upgrade pip 
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/requirements.txt 

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir /talker
COPY ./ /talker/

WORKDIR /talker
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000", "--timeout", "400"]