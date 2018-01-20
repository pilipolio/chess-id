import os

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from chessid import classifier
from chessid import detection


def image_to_annotated_squares(img):
    board = detection.find_board(img)
    squares = detection.split_board(board)
    square_size = squares[0].shape[0]

    model = classifier.ImageClassifier.load(
        num_classes=len(classifier.CLASSES),
        model_path=os.path.join(os.path.dirname(__file__), 'trained_non_lin_model_best.pth.tar'))

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
