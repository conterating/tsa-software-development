from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Load the model, specifying the 'mse' metric explicitly
model = load_model("model_2.h5", custom_objects={"mse": MeanSquaredError()})
#load dataset
df = pd.read_csv("crop_yield.csv")

#drop rows with na data
df.dropna(inplace=True)

#one hot encode data
encoded_df = pd.get_dummies(df, drop_first=True)

X = encoded_df.drop(["Yield_tons_per_hectare"], axis=1)
y = encoded_df["Yield_tons_per_hectare"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def make_prediction(data):
  prediction = model.predict(data)
  print(prediction)