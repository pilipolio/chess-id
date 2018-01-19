from flask import Flask, request, redirect, url_for, jsonify

import numpy as np

from flask import Flask, abort, send_file
from io import BytesIO

from chessid import annotator


app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    annotated_image = annotator.image_to_annotated_squares(img=np.asarray(bytearray(file.read())))
    byte_io = BytesIO()
    annotated_image.save(byte_io, 'PNG')
    byte_io.seek(0)
    return send_file(byte_io, mimetype='image/png')


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

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        app.run(host='0.0.0.0', debug=False)
    else:
        with open(sys.argv[1], 'rb') as file:
            img = np.asarray(bytearray(file.read()))
            annotated = annotator.image_to_annotated_squares(img)
            annotated.show()
            annotated.save('annotated.png')