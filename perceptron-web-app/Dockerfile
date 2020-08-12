FROM continuumio/miniconda3:latest

RUN conda install -y flask
RUN conda install -c pytorch  torchvision 

RUN mkdir -p /app
COPY . /app
EXPOSE 5070
CMD python ./app/nn.py