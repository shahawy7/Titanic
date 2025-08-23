import pickle
import pandas as pd
import requests


# test = pd.DataFrame([{
#     'Pclass': 3,
#     'Sex': 'male',
#     'Age': 22.0,
#     'SibSp': 1,
#     'Parch': 0,
#     'Fare': 7.25,
#     'Embarked': 'S'
# }])

while True:
    try:
        model_number = int(input("""
Choose a model
[1] Linear Regression
[2] Neural Network
[3] XGBClassifier                       
        """))
    except TypeError:
        print('Please enter a valid number')
        continue
    if model_number == 1:
        model = 'LinReg'
        break
    elif model_number == 2:
        model = 'NN'
        break
    elif model_number == 3:
        model = 'XGB'
        break
    else:
        print('Enter a valid number.')
        continue


test = []
cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
for i in cols:
    test.append(input(f'{i}: '))


data = {'model': model, 'features': [test]}

response = requests.post('http://127.0.0.1:3000/predict',
                         json=data, headers={'Content-Type': 'application/json'})
print(response.json())
