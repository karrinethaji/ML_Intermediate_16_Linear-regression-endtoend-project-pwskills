from flask import Flask, request, jsonify , render_template # jsonify is used to return the value in form of json (dictionay)
import pickle # to pickle our model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app= application

# Import ridge regression and sclaer objects for predicing the output based on the data

ridge= pickle.load(open("models/ridge.pkl","rb"))
scaler= pickle.load(open("models/scaler.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")
    # return render_template("home.html") # uncomment this and comment the above line and observe the output

@app.route("/predict_data",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="POST": # after clicking the submit button in the home.html, the data will be retrieved using POST method
        Temperature = float(request.form.get("temp"))
        RH = float(request.form.get("rh"))
        Ws = float(request.form.get("ws"))
        Rain = float(request.form.get("rain"))
        FFMC = float(request.form.get("ffmc"))
        DMC = float(request.form.get("dmc"))
        ISI = float(request.form.get("isi"))
        Classes = float(request.form.get("classes"))
        Region = float(request.form.get("region"))

        # sclae the new data using scaler object

        new_scaled_data= scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

        # predicting the output using scales data

        output=ridge.predict(new_scaled_data) # Output is in form of list 

        return render_template("home.html",result=output[0]) # Output value will be rendered in the home.html page

    else:
        return render_template("home.html") # It will render the home.html page

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)