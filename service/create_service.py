import flask
from torch import tensor, sigmoid


def create_service(model):
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

    return app
