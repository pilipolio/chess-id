import numpy as np
import unittest
from pathlib import Path

from chessid import detection


class Test(unittest.TestCase):

    def test_on_real_board(self):

        with Path(__file__).with_name('small_working.jpeg').open('rb') as f:
            image = np.asarray(bytearray(f.read()))

        debug_image, board_image, squares_images = detection.find_board(image)
        self.assertEqual(len(squares_images), 64)