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
        return flask.jsonify({'code': 200,'y_pred_prob': y_pred}), 200
    else:
        return flask.jsonify({'code': 400}), 400

@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.json['task'] == 'predict':
        x = tensor(flask.request.json['features'])
        y_pred = model.predict(x).flatten().data.numpy().tolist()
        return flask.jsonify({'code': 200,'y_pred': y_pred}), 200
    else:
        return flask.jsonify({'code': 400}), 400

if __name__ == '__main__':

    print("Initializing model server...")
    config = ParseServing.build_from_config()

    global model
    model = BasicNetPredict(config.model_path + '/' + config.model_name)


    #from cerberus import Validator
    #schema = {'name': {'type': 'string'}}
    #v = Validator(schema)

    #document = {'name': 'john doe'}

    #if v.validate(document): 
    #    print('data is valid')
    #else:
    #    print('invalid data')
    
    app.run(debug=True, port=config.port)


    #curl --header "Content-Type: application/json"  --request POST  --data '{"task":"predict_prob","features":[[2.505870783848182,-0.3987875040855265,-0.030373456741237377,-0.144803531037397,5.177408134507407,0.7729297452241753,1.0874247170744862,-0.7017820816643504,-0.3772831763958184,1.2154329838384652],[-1.5056913908365146,-0.49526208690811324,-0.030373456741237377,-0.144803531037397,-0.21712709919583073,-1.6481203808733793,-0.9196038900884225,1.424943762639813,-0.3772831763958184,-0.8227520672031581]]}'  http://localhost:5000/predict_prob
   #curl --header "Content-Type: application/json"  --request POST  --data '{"task":"predict","features":[[2.505870783848182,-0.3987875040855265,-0.030373456741237377,-0.144803531037397,5.177408134507407,0.7729297452241753,1.0874247170744862,-0.7017820816643504,-0.3772831763958184,1.2154329838384652],[-1.5056913908365146,-0.49526208690811324,-0.030373456741237377,-0.144803531037397,-0.21712709919583073,-1.6481203808733793,-0.9196038900884225,1.424943762639813,-0.3772831763958184,-0.8227520672031581]]}'  http://localhost:5000/predict 