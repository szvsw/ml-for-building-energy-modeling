# Use amazon/aws-lambda-python as a Base
# FROM public.ecr.aws/lambda/python:3.8
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
RUN pip install -r requirements.txt \
    && pip install -e ../archetypal \
    && pip install geomeppy \
    && pip install -U energy-pandas \
    && pip install awslambdaric

# Copy the current directory contents into the container at /app
COPY data/template_libs data/template_libs
COPY data/*.json data/
COPY data/*.csv data/
COPY data/epws/*.csv data/epws/
# COPY data/epws/*.npy data/epws/
COPY data/schedules.npy data/schedules.npy
COPY data/residential_schedules.npy data/residential_schedules.npy
COPY *.py .
COPY utils/*.py utils/
COPY shoeboxer/*.py shoeboxer/
COPY shoeboxer/*.json shoeboxer/
COPY api/lambda/eplus/shoeboxer/ api/lambda/eplus/shoeboxer/
  
ENV PYTHONPATH=/usr/src:$PYTHONPATH
# Run app.py when the container launches
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric"]
CMD ["api/lambda/eplus/shoeboxer/batch.handler"]
