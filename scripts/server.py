import sys
import os.path
import flask
from torch import tensor, sigmoid

# to add above path so that we can import built libraries
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                os.path.pardir)))

from src.model.predict.basicpredict import BasicNetPredict
from config import ParseServing

app = flask.Flask(__name__)


@app.route("/alive")
def hello():
    return "True", 200


@app.route("/predict_prob", methods=["POST"])
def predict_prob():

    if flask.request.json['task'] == 'predict_prob':
        x = tensor(flask.request.json['features'])
        y_pred = sigmoid(model.forward(x)).flatten().data.numpy().tolist()
        return flask.jsonify({'code': 200, 'y_pred_prob': y_pred}), 200
    else:
        return flask.jsonify({'code': 400}), 400


@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.json['task'] == 'predict':
        x = tensor(flask.request.json['features'])
        y_pred = model.predict(x).flatten().data.numpy().tolist()
        return flask.jsonify({'code': 200, 'y_pred': y_pred}), 200
    else:
        return flask.jsonify({'code': 400}), 400


if __name__ == '__main__':

    print("Initializing model server...")
    config = ParseServing.build_from_config()

    global model
    model = BasicNetPredict(config.model_path + '/' + config.model_name)

    app.run(debug=True, port=config.port)
