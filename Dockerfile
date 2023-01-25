FROM python:3.7

RUN mkdir code

WORKDIR /code

COPY . .

COPY ./requirements.txt /code/requirements.txt

RUN pip install -U pip wheel cmake

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

EXPOSE 8000

WORKDIR /code/app

CMD python -m uvicorn main:app --host 0.0.0.0 --port 8000