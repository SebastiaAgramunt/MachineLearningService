import sys
import os.path
import flask
from torch import tensor, sigmoid
import argparse
import ast
import requests
import json
# to add above path so that we can import built libraries
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
# os.path.pardir)))

from config import ParseFakeClient

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=str, default="0", help="Id of the client")


if __name__ == '__main__':

    args = parser.parse_args()
    config = ParseFakeClient.build_from_config()
    url = f'http://127.0.0.1:{config.port}/'

    print(f"Initializing client with id {args.id}...")
    try:
        r = requests.get(f'{url}alive')
    except requests.exceptions.RequestException as e:  # This is the correct syntax
        print(e)
        sys.exit(1)


    query1 = {"task": "predict_prob", "features":[]}
    with open(config.data_dir+"/processed/x_test_norm.csv") as file:
        for _ in range(10):
            features = file.readline().replace("\n","")
            features = list(ast.literal_eval(features))
            query1["features"].append(features)

    r = requests.post(f'{url}predict_prob', json=query1)

    print("Response to query of 10 features:")
    print(json.loads(r.content))

    with open(config.data_dir+"/processed/x_test_norm.csv") as file:
        for i in range(100):
            query1 = {"task": "predict_prob", "features":[]}
            features = file.readline().replace("\n","")
            features = list(ast.literal_eval(features))
            query1["features"].append(features)

            r = requests.post(f'{url}predict_prob', json=query1)

            print(f"Response to query of 1, {i} line:")
            print(json.loads(r.content))


    
