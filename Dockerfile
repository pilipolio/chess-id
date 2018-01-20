FROM python:3.6-slim
#FROM jjanzic/docker-python3-opencv
#RUN apt-get update && apt-get install -y python-opencv

WORKDIR /app
RUN apt-get update && apt-get -y install libxext6 libglib2.0-0 libsm6 libxrender1
ADD ./*requirements.txt /app/
RUN pip install -r requirements.txt
ADD . /app

ENTRYPOINT ["gunicorn", "-b", ":8080", "app:app", "--pythonpath", "src,src/chessid", "--capture-output"]