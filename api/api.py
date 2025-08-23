import flask
import pickle
from flask import request
import pandas as pd

with open('api/LogReg.pkl', 'rb') as file:
    LinReg = pickle.load(file)

with open('api/NN.pkl', 'rb') as file:
    NN = pickle.load(file)

with open('api/XGB.pkl', 'rb') as file:
    XGB = pickle.load(file)


app = flask.Flask(__name__)


@app.route("/predict", methods=['POST'])
def predict():
    model_num, features = request.get_json()['model'], request.get_json()[
        'features']
    cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    d = pd.DataFrame(features, columns=cols)
    if model_num == 'LinReg':
        prediction = LinReg.predict(d)
    elif model_num == 'NN':
        prediction = NN.predict(d)
    elif model_num == 'XGB':
        prediction = XGB.predict(d)

    response = {}
    response['prediction'] = prediction.tolist()
    return flask.jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, port=3000)
