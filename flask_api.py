from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger # automatically generates frontend UI

# 2 main APIs

# Starting a flask app
app=Flask(__name__)
# initialising app from Swagger
Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

# Provide a decorator
# Creating route API 
@app.route('/')
def welcome():
    return "Welcome All"

# Adding decorator
@app.route('/predict',methods=["Get"])
def predict_note_authentication():
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    # Comes from the pickle file 
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return "Hello The answer is"+str(prediction)

# Using predict file for testing 
# Method POST used 
@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction=classifier.predict(df_test)
    
    return str(list(prediction))

# When IT IS true
if __name__=='__main__':
    app.run(host='127.0.0.1',port=8000)
    
    