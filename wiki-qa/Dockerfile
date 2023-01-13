FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt update && DEBIAN_FRONTEND="noninteractive" apt install -y -qq make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl git llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    gnupg software-properties-common curl unzip openssh-client \
    apt-transport-https ca-certificates \
    zip

ENV HOME=/home
ENV PYENV_ROOT=$HOME/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT

ARG PYTHON_VERSION=3.8.16

RUN pyenv install $PYTHON_VERSION
RUN pyenv global $PYTHON_VERSION

WORKDIR /model

COPY . .

RUN pip install -r requirements.txt

RUN python setup.py


WORKDIR /container

COPY oML .

RUN pip install -r requirements.txt

RUN chmod +x /container/entrypoint.sh

ENTRYPOINT [ "/container/entrypoint.sh" ]
