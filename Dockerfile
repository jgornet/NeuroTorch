FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

WORKDIR /app

ADD . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -c pytorch -c conda-forge --yes --file requirements.txt && \
     /opt/conda/bin/conda clean -ya

EXPOSE 6006

ENV NAME neurotorch
