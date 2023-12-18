# Use an official Python runtime as a parent image
FROM python:3.8-slim

ENV TRAVIS_OS_NAME=linux
ENV ENERGYPLUS_VERSION=22.2.0
ENV ENERGYPLUS_SHA=c249759bad
ENV ENERGPYPLUS_TAG=v22.2.0
ENV ENERGYPLUS_INSTALL_VERSION=22-2-0

WORKDIR /usr/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget bzip2 ca-certificates curl git libxml2-dev sudo tk python3-tk libgomp1 \
    && git clone https://github.com/samuelduchesne/archetypal.git \
    && cd archetypal \
    && sudo chmod +x install_energyplus.sh  \
    && ./install_energyplus.sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /usr/src/ml-for-bem

COPY requirements/sampling-requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN pip install -e ../archetypal
RUN pip install geomeppy
RUN pip install -U energy-pandas

COPY idfworker/*.py idfworker/ 

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/ml-for-bem"



# Run app.py when the container launches
CMD ["python", "./idfworker/worker.py", "--queue", "ml-for-bem-idfworker", "--bucket", "ml-for-bem", "--experiment", "idf/batch/test", "--batch_id", "testbatch", "--num_messages_to_process", "3"]
