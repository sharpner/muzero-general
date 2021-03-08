FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

RUN apt-get update

ADD requirements.txt /
ADD games/ /games
ADD *.py /

RUN apt-get install -y python3-pip

RUN pip3 install -v -r requirements.txt

ENTRYPOINT ["python3", "muzero.py", "tictacnine"]
