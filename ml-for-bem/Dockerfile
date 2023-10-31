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

COPY sampling-requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN pip install -e ../archetypal
RUN pip install geomeppy
RUN pip install -U energy-pandas

# Copy the current directory contents into the container at /app
COPY data/template_libs data/template_libs
COPY data/*.json data/
# COPY data/epws/city_epws_indexed/*.epw data/epws/city_epws_indexed/
COPY data/epws/global_epws_indexed/*.epw data/epws/global_epws_indexed/ 
COPY data/*.csv data/
COPY data/epws/*.csv data/epws/
COPY data/epws/*.npy data/epws/
COPY data/schedules.npy data/schedules.npy
COPY data/residential_schedules.npy data/residential_schedules.npy
COPY *.py .
COPY utils/*.py utils/
COPY shoeboxer/*.py shoeboxer/
COPY shoeboxer/*.json shoeboxer/
  
# Set the working directory in the container
WORKDIR /usr/src/ml-for-bem




# Run app.py when the container launches
CMD ["python", "full_climate_seed.py", "--n", "10", "--bucket", "ml-for-bem", "--experiment_name", "full_climate_zone/docker-test/train", "--train_or_test","train"]
