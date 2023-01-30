FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ENV PORT=8000

RUN pip install sentence-transformers fastapi uvicorn

WORKDIR /model

COPY init.py .

RUN python init.py

COPY . .

RUN chmod +x ./entrypoint.sh

ENTRYPOINT [ "/model/entrypoint.sh" ]
