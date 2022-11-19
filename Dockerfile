FROM python:3

WORKDIR .

COPY zero_deforestation ./zero_deforestation
COPY requirements.txt .
COPY Makefile .

RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT bash
