FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /model

COPY . .

RUN pip install sentence-transformers

RUN python init.py


WORKDIR /container

COPY oML .

RUN pip install -r requirements.txt

RUN chmod +x /container/entrypoint.sh

ENTRYPOINT [ "/container/entrypoint.sh" ]
