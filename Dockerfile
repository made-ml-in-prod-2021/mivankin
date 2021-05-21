FROM python:3.9-slim

COPY online_inference ./online_inference

WORKDIR /online_inference

RUN apt-get update
RUN apt-get -y install gcc
RUN pip install --upgrade pip
RUN pip install scipy
RUN pip install -r requirements.txt
	
WORKDIR /online_inference/fast_api

ENV PYTHONPATH "{PYTHONPATH}:/online_inference/src:/online_inference"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]