FROM python:3.10-slim

WORKDIR /TalkToPDF

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY htmlTemplates.py .

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]