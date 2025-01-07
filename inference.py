import joblib
import os
import numpy as np
from sagemaker.sklearn import SKLearnModel
from sagemaker import get_execution_role

# Define inference logic
def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'iris_model.joblib')
    model = joblib.load(model_path)
    return model

def predict(input_data):
    model = model_fn('/opt/ml/model')
    prediction = model.predict(np.array(input_data).reshape(1, -1))
    return prediction

