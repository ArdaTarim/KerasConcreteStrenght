from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# importing the dataset
df = pd.read_csv("src/concrete_data.csv")
print(df.head)

# creating X and Y datasets
X = df.drop(columns=["Strength"])
Y = df["Strength"]

# splitting them into training and testing datasets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42)

# normalizing the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# creating the model with one hidden layer with 10 nodes
model = Sequential()
no_cols = X_train.shape[1]
model.add(Dense(10, activation="relu", input_dim=no_cols))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mean_squared_error")

mse_list = []


def train_model_and_report_error() -> float:
    # training the model with training data
    model.fit(X_train, Y_train, epochs=100)

    # making the predictions
    predictions = model.predict(X_test)

    # evaluating the model
    mse = mean_squared_error(Y_test, predictions)
    mse_list.append(mse)
    print(f'Mean Squared Error: {mse}')
    return mse


erros = []
for i in range(50):
    erros.append(train_model_and_report_error())

mean_of_erros = np.mean(erros)
standart_deviation_of_erros = np.std(erros)

print(
    f"After 50 iterations with normalized data and 100 epochs \nMean of the data is: {mean_of_erros}\nStandart Deviation: {standart_deviation_of_erros}")

# After 50 iterations with normalized data and 100 epochs
# Mean of the error data is: 41.72675271724622
# Standart Deviation of the error data is: 17.313033521974475

graph_data_3 = pd.DataFrame(mse_list)

graph_data_3.to_csv("proj3_graph", index=False)
