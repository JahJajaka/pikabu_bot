FROM nvcr.io/nvidia/tensorrt:21.06-py3

COPY requirements.txt /tmp/

RUN apt-get update \
    && apt-get install -y python3-psycopg2 \
    && apt-get install -y gettext-base \
    && apt-get install -y bash \
    && apt-get install -y libncurses5 \
    && rm -rf /var/lib/apt/lists
RUN pip install --upgrade pip 
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/requirements.txt 

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir /talker
COPY ./ /talker/

WORKDIR /talker
#COPY inference_models/gpt2_helper_opset12.py /opt/conda/lib/python3.8/site-packages/onnxruntime/transformers/gpt2_helper.py
#COPY inference_models/convert_graph_to_onnx_v2.py /opt/conda/lib/python3.8/site-packages/transformers/convert_graph_to_onnx.py
#COPY inference_models/onnx /opt/conda/lib/python3.8/site-packages/transformers/onnx
#COPY inference_models/onnx_utils.py /opt/conda/lib/python3.8/site-packages/torch/onnx/utils.py
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD tail -f /dev/null
#CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000", "--timeout", "400"]

