FROM python:3.9

WORKDIR /usr/src

COPY requirements/ubem-requirements.txt ubem-requirements.txt
COPY requirements/api-requirements.txt api-requirements.txt

RUN pip install -r ubem-requirements.txt -r api-requirements.txt

COPY umi/*.py umi/
COPY weather/*.py weather/
COPY api/fast/*.py api/fast/
COPY shoeboxer/*.py shoeboxer/
COPY utils/*.py utils/
COPY *.py ./

ENV PYTHONPATH=/usr/src:$PYTHONPATH
EXPOSE 80

CMD ["uvicorn", "api.fast.main:app", "--host", "0.0.0.0", "--port", "80"]