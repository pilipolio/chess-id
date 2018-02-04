import os

from PIL.Image import Image
from PIL import ImageFont
from PIL import ImageDraw
import numpy as np

from chessid import classifier

APP_ENGINE_NAME = 'chess-id.appspot.com'
DEFAULT_MODEL_NAME = 'trained_non_lin_model_best.pth.tar'

model = None


def load_model():
    global model
    if model is not None:
        print('model exists')
        return

    if 'MODEL_PATH' in os.environ and os.path.exists(os.environ.get('MODEL_PATH')):
        model_path = os.environ.get('MODEL_PATH')
        print(f'Using model found at {model_path}')
    else:
        print('loading model from gc storage')
        from google.cloud import storage
        client = storage.Client()
        bucket = client.get_bucket(APP_ENGINE_NAME)
        blob = bucket.get_blob(DEFAULT_MODEL_NAME)
        model_path = DEFAULT_MODEL_NAME
        blob.download_to_filename(DEFAULT_MODEL_NAME)
        print(f'{model_path} downloaded')

    model = classifier.ImageClassifier.load(
        num_classes=len(classifier.CLASSES),
        model_path=model_path)


load_model()


from typing import NamedTuple, List

Annotations = NamedTuple('Annotations', [
    ('squares', List[Image]),
    ('grid', List[List[str]])
])


def annotated_squares(squares_images: List[Image]) -> Annotations:

    grid = np.empty(shape=(8, 8, 13), dtype="<U10")

    for i, square_image in enumerate(squares_images):
        row_index, col_index  = int(i / 8), int(i % 8)

        image_tensor = classifier.pil_image_to_tensor(square_image)
        print(image_tensor.shape)
        pred = model(image_tensor)

        ordered_indexes = pred.topk(13)[1].data.numpy().flatten()
        ordered_labels = np.asarray(classifier.CLASSES)[ordered_indexes]

        print(ordered_labels[0])
        grid[row_index, col_index] = ordered_labels

    return Annotations(
        squares=squares_images,
        grid=grid
    )


def draw_label(pil_img, text):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/SFNSText.ttf', 40)
    except:
        print('using default fixed size font...')
        font = ImageFont.load_default()
    draw.text((0, 0), text, fill=(255,0, 0), font=font)