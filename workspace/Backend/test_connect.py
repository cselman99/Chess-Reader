from flask import Flask
from flask_testing import TestCase
from cv2 import imread

FILENAME = '../Training/board4.png'
IMG_UPLOAD_SUCESS = 1

class TestViews(TestCase):

    def create_app(self):
        app = Flask(__name__)
        app.config['TESTING'] = True
        return app

    def test_upload_file(self):
        img = imread(FILENAME)
        response = self.client.post(img)
        assert(response == IMG_UPLOAD_SUCESS)
