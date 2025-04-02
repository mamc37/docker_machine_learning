FROM python:3.10

WORKDIR /app
COPY docker.py /app/docker.py
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
CMD ["python", "./docker.py"]