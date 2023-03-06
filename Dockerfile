FROM ubuntu:20.04
RUN apt-get update && apt install -y python3-dev python3-pip python3.8.14-venv
ENV PATH = "sentiment_env/bin:${PATH}"
RUN python3 -m venv sentiment_env
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
EXPOSE 8080
WORKDIR /app
COPY 