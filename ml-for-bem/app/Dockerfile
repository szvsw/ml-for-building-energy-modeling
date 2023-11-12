FROM python:3.9

WORKDIR /usr/src

COPY requirements/ubem-requirements.txt ubem-requirements.txt
COPY requirements/app-requirements.txt app-requirements.txt
COPY requirements/plot-requirements.txt plot-requirements.txt

RUN pip install -r ubem-requirements.txt -r app-requirements.txt -r plot-requirements.txt

COPY umi/*.py umi/
COPY weather/*.py weather/
COPY app/*.py app/
COPY shoeboxer/*.py shoeboxer/
COPY utils/*.py utils/
COPY *.py ./

ENV PYTHONPATH=/usr/src:$PYTHONPATH
EXPOSE 8501

CMD ["streamlit", "run", "app/main.py", "--server.port", "8501", "--server.address", "0.0.0.0"]