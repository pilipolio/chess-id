import datetime as dt
import json
import os
from pathlib import Path
import shutil

from flask import Flask, request, render_template, url_for
import numpy as np

from chessid import annotator
from chessid import classifier

app = Flask(__name__)
module_directory = Path(__file__).parent


def get_data_directory() -> Path:
    data_directory = Path(os.environ.get('DATA_DIRECTORY', 'data'))
    for label in classifier.CLASSES:
        square_dir = data_directory.joinpath('squares', label.lower() or 'empty')
        if not square_dir.exists():
            square_dir.mkdir(parents=True)

    boards_dir = data_directory.joinpath('boards')
    if not boards_dir.exists():
        boards_dir.mkdir()

    return data_directory


def square_image_path(image_id, index) -> Path:
    return module_directory.joinpath('static', 'squares', f'{image_id}_{index}.png')


@app.route('/upload', methods=['POST'])
def upload_file():
    image_id = f'{dt.datetime.now():%Y%m%d%H%M%s}'
    file = request.files['file']
    from chessid import detection
    detection_result = detection.find_board(np.asarray(bytearray(file.read())))

    if detection_result.board is None:
        static_relative_path = Path('edges') / Path(f'{image_id}.png')
        with (module_directory.joinpath('static') / static_relative_path).open(mode='wb') as f:
            detection_result.debug.save(f, 'PNG')

        print('return debug')
        return render_template(
            'debug.html',
            debug_image_path=url_for('static', filename=static_relative_path),
        )

    annotation = annotator.annotated_squares(detection_result.squares)

    static_relative_path = Path('boards') / Path(f'{image_id}.png')
    with (module_directory.joinpath('static') / static_relative_path).open(mode='wb') as f:
        detection_result.board.save(f, 'PNG')

    for index, square_image in enumerate(annotation.squares):
        with square_image_path(image_id, index).open(mode='wb') as f:
            square_image.save(f, 'PNG')

    return render_template(
        'annotations.html',
        annotated_path=url_for('static', filename=static_relative_path),
        grid=annotation.grid.tolist(),
        image_id=image_id
    )


@app.route('/upload', methods=['GET'])
def upload_form():
    return '''
    <!doctype html>
    <title>Chess ID</title>
    <h1>Upload board picture</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


@app.route('/annotations/<image_id>', methods=['POST'])
def post_pieces(image_id):
    data_directory_path = get_data_directory()

    print(f'saving annotations for {image_id} and copying to {data_directory_path}')
    pieces = request.get_json(force=True)['pieces']

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
    app.run(port=5001)
