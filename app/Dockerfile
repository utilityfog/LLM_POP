FROM python:3.11-slim

ENV PORT 8080

RUN apt update && apt install -y libpq-dev

# Copy only the requirements.txt to the container working directory first to leverage Docker cache
COPY ./requirements.txt /requirements.txt

# Install dependencies; using requirements instead of poetry
RUN pip install --no-cache-dir -r requirements.txt
COPY ./source_docs ./source_docs
COPY ./app ./app

EXPOSE $PORT

CMD exec uvicorn app.server:app --host 0.0.0.0 --port $PORT