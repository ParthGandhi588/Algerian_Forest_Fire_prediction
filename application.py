from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

appilication = Flask(__name__)
app=appilication

## importing the model
ridge_model = pickle.load(open('model/ridge.pkl','rb'))
standard_scaler = pickle.load(open('model/scaler.pkl','rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperatur = float(request.form.get('temperature'))
        RH = float(request.form.get('rh'))
        Ws = float(request.form.get('ws'))
        Rain = float(request.form.get('rain'))
        FFMC = float(request.form.get('ffmc'))
        DMC = float(request.form.get('dmc'))
        ISI = float(request.form.get('isi'))
        Classes = float(request.form.get('classes'))
        Region = float(request.form.get('region'))

        new_data_scaled = standard_scaler.transform([[Temperatur,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html',results=result[0])

    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)