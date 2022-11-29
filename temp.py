# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    TargetUser : str
    ProductType : str
    Condition1 : str
    Condition2 : str
    Condition3 : str
    Condition4 : str
    Condition5 : str
    
product_model = pickle.load(open('trained_model.sav','rb'))

@app.post('/product_recommendation')
def product_pred(input_parameters : model_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    tu = input_dictionary['TargetUser']
    pt = input_dictionary['ProductType']
    c1 = input_dictionary['Condition1']
    c2 = input_dictionary['Condition2']
    c3 = input_dictionary['Condition3']
    c4 = input_dictionary['Condition4']
    c5 = input_dictionary['Condition5']
    
    input_list = [tu, pt, c1, c2, c3, c4, c5]
    
    prediction = product_model.predict([input_list])

    return prediction    
    
