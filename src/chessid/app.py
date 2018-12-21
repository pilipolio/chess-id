import datetime as dt
import json
import os
from pathlib import Path
import shutil

from flask import Flask, request, render_template, url_for
import numpy as np

from chessid import detection
from chessid import annotator
from chessid import classifier

STATIC_URL_PATH = Path('/tmp/chess-id')
STATIC_URL_PATH.joinpath('squares', 'train').mkdir(parents=True, exist_ok=True)
STATIC_URL_PATH.joinpath('edges').mkdir(parents=True, exist_ok=True)
STATIC_URL_PATH.joinpath('boards').mkdir(parents=True, exist_ok=True)
shutil.copy(Path(__file__).parent.joinpath('static/annotate.js').as_posix(), STATIC_URL_PATH.as_posix())
shutil.copy(Path(__file__).parent.joinpath('static/jquery-3.2.1.min.js').as_posix(), STATIC_URL_PATH.as_posix())

app = Flask(__name__, static_folder=STATIC_URL_PATH.as_posix())


def get_data_directory() -> Path:
    data_directory = Path(os.environ.get('DATA_DIRECTORY', 'data'))
    for label in classifier.CLASSES:
        square_dir = data_directory.joinpath('squares', label.lower() or 'empty')
        square_dir.mkdir(parents=True, exist_ok=True)

    data_directory.joinpath('boards').mkdir(exist_ok=True)

    return data_directory


def square_image_path(image_id, index) -> Path:
    return STATIC_URL_PATH.joinpath('squares', f'{image_id}_{index}.png')


@app.route('/debug', methods=['POST'])
def debug():
    image_id = f'{dt.datetime.now():%Y%m%d%H%M%s}'
    file = request.files['file']
    detection_result = detection.find_board(np.asarray(bytearray(file.read())))
    return debug_page(image_id, detection_result)


def debug_page(image_id, detection_result: detection.Result):
    relative_path = Path('edges') / Path(f'{image_id}.png')
    with STATIC_URL_PATH.joinpath(relative_path).open(mode='wb') as f:
        detection_result.debug.save(f, 'PNG')
    return render_template(
        'debug.html',
        debug_image_path=url_for('static', filename=relative_path),
    )


@app.route('/upload', methods=['POST'])
def upload_file():
    image_id = f'{dt.datetime.now():%Y%m%d%H%M%s}'
    file = request.files['file']
    detection_result = detection.find_board(np.asarray(bytearray(file.read())))

    if detection_result.board is None:
        return debug_page(image_id, detection_result)

    annotation = annotator.annotated_squares(detection_result.squares)

    relative_path = Path('boards').joinpath(f'{image_id}.png')
    with STATIC_URL_PATH.joinpath(relative_path).open(mode='wb') as f:
        detection_result.board.save(f, 'PNG')

    for index, square_image in enumerate(annotation.squares):
        with square_image_path(image_id, index).open(mode='wb') as f:
            square_image.save(f, 'PNG')

    return render_template(
        'annotations.html',
        annotated_path=url_for('static', filename=relative_path),
        grid=annotation.grid.tolist(),
        image_id=image_id
    )


@app.route('/upload', methods=['GET'])
def upload_form():
    return render_template('upload.html')


@app.route('/annotations/<image_id>', methods=['POST'])
def post_pieces(image_id):
    data_directory_path = get_data_directory()

    print(f'saving annotations for {image_id} and copying to {data_directory_path}')
    pieces = request.get_json(force=True)['pieces']
    STATIC_URL_PATH.joinpath('boards').joinpath(f'{image_id}.json').write_text(json.dumps(pieces))
    for index, label in enumerate(pieces):
        print(index, label)
        square_path = square_image_path(image_id, index)
        target_path = data_directory_path.joinpath('squares', 'train', label.lower() or 'empty', square_path.parts[-1])
        shutil.copy(square_path, target_path)

    return 'ok'


@app.route('/_ah/health', methods=['GET'])
def health():
    return '', 200


if __name__ == '__main__':
    app.run()
