import os

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from chessid import classifier
from chessid import detection

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


def image_to_annotated_squares(img):
    board = detection.find_board(img)
    print(board.shape)
    if board.shape[0] != board.shape[1]:
        return Image.fromarray(board)


    squares = detection.split_board(board)
    square_size = squares[0].shape[0]

    annotated_board = Image.new('RGBA', (square_size * 8, square_size  * 8))

    for i, square in enumerate(squares):
        image = classifier.cv_to_pil(square)
        image_tensor = classifier.pil_image_to_tensor(image)
        print(image_tensor.shape)
        pred = model(image_tensor)
        print(pred)

        top_index = int(pred.topk(1)[1].data[0][0])
        label = classifier.CLASSES[top_index]
        print(label)

        draw_label(image , label)
        offset = int(i % 8) * square_size, int(i / 8) * square_size
        annotated_board.paste(image , offset)

    return annotated_board


def draw_label(pil_img, text):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/SFNSText.ttf', 40)
    except:
        print('using default fixed size font...')
        font = ImageFont.load_default()
    draw.text((0, 0), text, fill=(255,0, 0), font=font)
