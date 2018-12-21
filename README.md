# Chess ID

This is a fork of [chess-id](https://github.com/daylen/chess-id), a chess board & pieces recognition server from bird's eye photo, with support for:
 * Pytorch instead of Caffe
 * Full instructions and requirements to run your own server locally (Docker or your Python3 of choice), or on Google cloud (see https://chess-id.appspot.com/upload)
 
 See https://medium.com/@daylenyang/building-chess-id-99afa57326cd for the original blog post
 
## Experiment with your own models

First, grab the data: https://www.dropbox.com/s/618l4ddoykotmru/Chess%20ID%20Public%20Data.zip?dl=0

Experiments are available in the Jupyter notebook.

## Download pre-trained model

you can download this pre-trained (far from perfect) model

```
$ wget https://storage.googleapis.com/chess-id.appspot.com/trained_non_lin_model_best.pth.tar
```

## Fine tune model


```
$ PYTHONPATH=src python  src/chessid/train.py --pretrained -a alexnet --lr 0.01 --batch-size 64 --epochs 3 data/squares --resume trained_non_lin_model_best.pth.tar
```


## Deploy the Chess ID server locally

```
$ pip install -r local-requirements.txt
$ MODEL_PATH=~/Data/trained_non_lin_model_best.pth.tar PYTHONPATH=src python src/chessid/app.py
```

or with docker

```
docker build -t chess-id .; docker run -e MODEL_PATH=trained_non_lin_model_best.pth.tar chess-id -e GOOGLE_APPLICATION_CREDENTIALS="LINK_TO_GCP_JSONchess-id-c567f2e698cd.json"
```

## Deploy the Chess ID server on Google App Engine

The server is deployed on [Google flexible app engine](https://cloud.google.com/appengine/docs/flexible), see https://chess-id.appspot.com/upload.
The environment is defined in the `app.yaml` and the `Dockerfile` and can be deployed by doing:

```
$ gcloud app deploy
```

and then logs can be remotely inspected with:

```
$ gcloud app logs tail
```

```
gcloud app versions list
gcloud app versions stop 
```