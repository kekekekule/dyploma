FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

ENV DEBIAN_FRONTEND noninteractive
RUN useradd -ms /bin/bash --uid 1000 jupyter\
 && apt update\
 && apt install -y python3.8-dev python3.8-distutils curl\
 && ln -s /usr/bin/python3.8 /usr/local/bin/python3\
 && curl https://bootstrap.pypa.io/get-pip.py | python3

COPY . .

RUN pip install --upgrade pip
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html

USER jupyter
