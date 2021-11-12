from flask import Flask, request
from flask_restx import Resource, Api, reqparse, fields
from joblib import load
from jsonschema.validators import validate
import numpy as np

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument("image", required=True, type=list, location='json',
                    help='A list of 64 integers representing a flattened 8x8 image.')
parser_model = api.model('predict', {'image': fields.List(fields.Float)})

# Load best model
model = load('../models/best_model.jolib')


@api.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


@api.route('/predict', methods=['POST', 'GET'])
class Perdict(Resource):
    def get(self):
        return "<p>This only supports POST requests with JSON data having 'image' key with flattened 8x8 array of numbers. (i.e. array of 64 numbers)</p>"

    def _process_image(self, image):
        return np.array(image).reshape(1, -1)

    @api.expect(parser_model)
    def post(self):
        args = parser.parse_args()
        if 'image' not in args:
            return f"<p>No image data provided.</p>"
        image = self._process_image(args['image'])
        try:
            pred = model.predict(image)
        except Exception as e:
            return f"Error: {e}"
        print(f"Input image: {image.shape}")
        print(f"Prediction: {pred}")
        return f'<p>Prediction: {pred}</p>'


if __name__ == '__main__':
    app.run(debug=True)
