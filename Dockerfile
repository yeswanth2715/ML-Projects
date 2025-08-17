from python:3.9-slim-buster
# Install dependencies
WORKDIR /app
COPY . /app
RUN apt update -y && apt install awscli -y && \

RUN pip install -r requirements.txt
CMD ["python3", "app.py"]