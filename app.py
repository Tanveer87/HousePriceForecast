# Importing Required Libraries and Modules
import os
import cv2
import base64
import pickle
import shutil
import requests
import datetime
import warnings
import numpy as np
import tensorflow as tf
from url_dict import url_dict
from urllib.request import urlopen
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, redirect, url_for, request

# Suppress warnings
warnings.filterwarnings('ignore')

# Instantiating the Flask App
app = Flask(__name__)


##### Routes for Root Pages #####


# Home Page
@app.route("/", methods=['GET'])
@app.route("/home", methods=['GET'])
def home():
    return render_template("home.html")

# About Page
@app.route("/about", methods=['GET'])
def about():
    return render_template("about.html")

# Machine Learning Page
@app.route("/machine-learning", methods=['GET'])
def machine_learning():
    return render_template("machine_learning.html")

# Deep Learning Page
@app.route("/deep-learning", methods=['GET'])
def deep_learning():
    return render_template("deep_learning.html")

# House Price Prediction Page
@app.route("/house-price-prediction", methods=['GET', 'POST'])
def house_price_prediction():
    return render_template("predict/house_price_prediction.html", data_dict=url_dict['1'])


#################################
##### Routes for Prediction #####
#################################

# House Price Prediction
@app.route("/house-price-prediction/predict", methods=['GET', 'POST'])
def house_price_prediction_predict():
    if request.method == 'POST':
        minmax_scaler_url = "https://the-ml-dl-app-bucket.s3.amazonaws.com/saved_models/house_price_prediction/minmax_scaler.pkl"
        label_encoder_url = "https://the-ml-dl-app-bucket.s3.amazonaws.com/saved_models/house_price_prediction/label_encoders.pkl"

        minmax_scaler = pickle.load(urlopen(minmax_scaler_url))
        label_encoder_dict = pickle.load(urlopen(label_encoder_url))

        GrLivArea = np.log(float(request.form['GrLivArea']))
        stFlrSF = np.log(float(request.form['1stFlrSF']))
        YearRemodAdd = int(request.form['YrSold']) - int(request.form['YearRemodAdd'])
        MSZoning = request.form['MSZoning']
        PavedDrive = request.form['PavedDrive']
        BldgType = request.form['BldgType']
        Fireplaces = int(request.form['Fireplaces'])
        KitchenQual = request.form['KitchenQual']
        GarageType = request.form['GarageType']
        GarageFinish = request.form['GarageFinish']
        GarageCars = int(request.form['GarageCars'])
        CentralAir = request.form['CentralAir']
        BsmtFullBath = int(request.form['BsmtFullBath'])
        BsmtExposure = request.form['BsmtExposure']
        BsmtFinType1 = request.form['BsmtFinType1']
        BsmtQual = request.form['BsmtQual']
        HeatingQC = request.form['HeatingQC']
        ExterQual = request.form['ExterQual']
        LotShape = request.form['LotShape']
        OverallQual = int(request.form['OverallQual'])
        Algorithm = request.form['Algorithm']

        if MSZoning == 'C (all)':
            MSZoning = 'Rare_Var'
        if LotShape == 'IR3':
            LotShape = 'Rare_Var'
        if ExterQual == 'Fa':
            ExterQual = 'Rare_Var'
        if HeatingQC == 'Po':
            HeatingQC = 'Rare_Var'
        if GarageType == 'CarPort' or GarageType == '2Types':
            GarageType = 'Rare_Var'

        MSZoning = label_encoder_dict['MSZoning'].transform([MSZoning])[0]
        LotShape = label_encoder_dict['LotShape'].transform([LotShape])[0]
        BldgType = label_encoder_dict['BldgType'].transform([BldgType])[0]
        ExterQual = label_encoder_dict['ExterQual'].transform([ExterQual])[0]
        BsmtQual = label_encoder_dict['BsmtQual'].transform([BsmtQual])[0]
        BsmtExposure = label_encoder_dict['BsmtExposure'].transform([BsmtExposure])[0]
        BsmtFinType1 = label_encoder_dict['BsmtFinType1'].transform([BsmtFinType1])[0]
        HeatingQC = label_encoder_dict['HeatingQC'].transform([HeatingQC])[0]
        CentralAir = label_encoder_dict['CentralAir'].transform([CentralAir])[0]
        KitchenQual = label_encoder_dict['KitchenQual'].transform([KitchenQual])[0]
        GarageType = label_encoder_dict['GarageType'].transform([GarageType])[0]
        GarageFinish = label_encoder_dict['GarageFinish'].transform([GarageFinish])[0]
        PavedDrive = label_encoder_dict['PavedDrive'].transform([PavedDrive])[0]

        X_test = [[MSZoning, LotShape, BldgType, OverallQual, YearRemodAdd, ExterQual, BsmtQual, BsmtExposure,
                   BsmtFinType1, HeatingQC, CentralAir , stFlrSF, GrLivArea, BsmtFullBath, KitchenQual, Fireplaces,
                   GarageType, GarageFinish, GarageCars, PavedDrive]]

        X_test = minmax_scaler.transform(X_test)

        if Algorithm == 'linear_regressor':
            model_url = "https://the-ml-dl-app-bucket.s3.amazonaws.com/saved_models/house_price_prediction/linear_regression_model.pkl"
        elif Algorithm == 'support_vector_regressor':
            model_url = "https://the-ml-dl-app-bucket.s3.amazonaws.com/saved_models/house_price_prediction/support_vector_regression_model.pkl"
        else:
            model_url = "https://the-ml-dl-app-bucket.s3.amazonaws.com/saved_models/house_price_prediction/random_forest_regression_model.pkl"

        model = pickle.load(urlopen(model_url))
        prediction = model.predict(X_test)[0]

        prediction_text = "The estimated price of the house is ${}".format(int(np.exp(prediction)))
        return render_template("predict/house_price_prediction.html", data_dict=url_dict['1'],
                                prediction_text=prediction_text, scroll='OutputPredictionText')
    return redirect(url_for('house_price_prediction'))


# Initializing the Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    # app.run(debug=True)