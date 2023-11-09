FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /usr/src
  
COPY requirements/runpod-requirements.txt requirements.txt
RUN pip install -r requirements.txt 

# TODO: should have a better mechanism of selecting model to load
COPY data/model-with-transform-configs.ckpt data/
COPY data/schedules.npy data/
COPY api/test_input.json .
COPY api/*.py api/
COPY ml/*.py ml/
COPY shoeboxer/*.py shoeboxer/
COPY utils/*.py utils/
COPY *.py ./

ENV PYTHONPATH=/usr/src:$PYTHONPATH

CMD [ "python", "-u", "api/handler.py" ]
