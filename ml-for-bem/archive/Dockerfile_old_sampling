# Use an official Python runtime as a parent image
FROM python:3.8-slim

ENV TRAVIS_OS_NAME=linux
ENV ENERGYPLUS_VERSION=9.2.0

WORKDIR /usr/src

# Get dependency libs
# clone repos
# install energy plus
    # cleanup
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget bzip2 ca-certificates curl git libxml2-dev sudo tk python3-tk \
    && git clone https://github.com/zlehong/pyumi.git \
    && git clone https://github.com/samuelduchesne/archetypal.git \
    && cd archetypal \
    && git checkout 775721f9a6f1451a9fca828a38e5e75874538d5e \
    && sudo chmod +x install_energyplus.sh  \
    && ./install_energyplus.sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /usr/src/ml-for-bem

COPY sampling-requirements.txt requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN pip install -e ../pyumi --no-deps
RUN pip install -e ../archetypal
RUN pip install geomeppy

# Copy the current directory contents into the container at /app
COPY data/template_libs data/template_libs
COPY data/*.json data/
COPY data/epws/city_epws_indexed/*.epw data/epws/city_epws_indexed/
COPY data/epws/*.npy data/epws/
COPY *.py .
  
# Set the working directory in the container
WORKDIR /usr/src/ml-for-bem




# Run app.py when the container launches
CMD ["python", "aws-batch-tester.py", "--mode", "test", "--name", "test-batch", "--artifact", "test-batch-artifact", "--job_id", "0", "--job_offset", "0", "--context", "AWS"]
