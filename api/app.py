from flask import Flask, request
from flask_restx import Resource, Api, reqparse
from joblib import load
import numpy as np

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument("image", required=True, type=list,
                    help='A list of 64 integers representing a flattened 8x8 image.')


@api.route('/hello')
class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


class Model(Resource):
    def __init__(self, model, *a, **kw) -> None:
        self._model = model
        super().__init__(*a, **kw)

    def get(self):
        return "<p>This only supports POST requests with JSON data having 'image' key with flattened 8x8 array of numbers. (i.e. array of 64 numbers)</p>"

    def _process_image(self, image):
        im = np.array(image)
        return im.reshape(1, -1)

    # @api.expect(parser_model)
    def post(self):
        args = request.json
        if 'image' not in args:
            return f"<p>No image data provided.</p>"
        image = self._process_image(args['image'])
        try:
            pred = self._model.predict(image)
        except Exception as e:
            print(e)
            return f"{{'Error': {e}}}"
        # Reutrn JSON
        return {'prediction': int(pred)}


@api.route('/svm_predict', methods=['POST', "GET"])
class SVM_Pred(Model):
    def __init__(self, *a, **kw) -> None:
        model = load('../models/best_8x8/SVC.gz')['model']
        super().__init__(model, *a, **kw)


@api.route('/decision_tree_predict', methods=['POST', "GET"])
class DCC_Pred(Model):
    def __init__(self, *a, **kw) -> None:
        model = load('../models/best_8x8/DecisionTreeClassifier.gz')['model']
        super().__init__(model, *a, **kw)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
