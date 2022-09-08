
from flask import Flask,request,app,jsonify,url_for,render_template

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import FloatType
import pickle
import json
import pandas as pd
import numpy as np


app=Flask(__name__)
## Load the model

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# Unpickle, pkl file
regmodel_pkl = sc.binaryFiles("regmodel.pkl")
model_rdd_data = regmodel_pkl.collect()

# Load and broadcast python object over spark nodes
regmodel = pickle.loads(model_rdd_data[0][1])
broadcast_reg_model = sc.broadcast(regmodel)

# Unpickle, pkl file - Scaler Model
scaler_rdd_pkl = sc.binaryFiles("scaling.pkl")
scaler_rdd_data = scaler_rdd_pkl.collect()

# Load and broadcast python object over spark nodes
scaler_model = pickle.loads(scaler_rdd_data[0][1])
broadcast_scaler = sc.broadcast(scaler_model)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/spark_predict_api',methods=['POST'])
def spark_predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    #Scaling 
    new_data=broadcast_scaler.value.transform(np.array(list(data.values())).reshape(1,-1))
    #Predict
    output=broadcast_reg_model.value.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/spark_predict',methods=['POST'])
def spark_predict():
    data=[float(x) for x in request.form.values()]
    #Scaling
    final_input=broadcast_scaler.value.transform(np.array(data).reshape(1,-1))
    #Predict
    print(final_input)
    output=broadcast_reg_model.value.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

if __name__=="__main__":
    app.run(debug=True)