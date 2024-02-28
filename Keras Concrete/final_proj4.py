from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# importing the dataset
df = pd.read_csv("concrete_data.csv")
print(df.head)

# creating X and Y datasets
X = df.drop(columns= ["Strength"])
Y = df["Strength"]

# splitting them into training and testing datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# normalizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# creating the model with one hidden layer with 10 nodes
model = Sequential()
no_cols = X_train.shape[1]
model.add(Dense(10, activation= "relu", input_dim = no_cols))
model.add(Dense(10, activation= "relu"))
model.add(Dense(10, activation= "relu"))
model.add(Dense(1))
model.compile(optimizer= "adam", loss= "mean_squared_error")

def train_model_and_report_error() -> float:
    # training the model with training data
    model.fit(X_train, Y_train, epochs= 50)

    # making the predictions 
    predictions = model.predict(X_test)

    # evaluating the model 
    mse = mean_squared_error(Y_test, predictions )
    print(f'Mean Squared Error: {mse}')
    return mse

erros = []
for i in range(50):
    erros.append(train_model_and_report_error())

mean_of_erros = np.mean(erros)
standart_deviation_of_erros = np.std(erros)

print(f"After 50 iterations with normalized data and 50 epochs with 3 hidden layers\nMean of the data is: {mean_of_erros}\nStandart Deviation: {standart_deviation_of_erros}")

# After 50 iterations with normalized data and 50 epochs with 3 hidden layers
# Mean of the data is: 34.23940513662155
# Standart Deviation: 12.238098475823424
