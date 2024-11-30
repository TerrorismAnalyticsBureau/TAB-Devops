import argparse

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import mlflow
import mlflow.sklearn

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import os
from datetime import datetime, timedelta
import calendar
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import json

TARGET_COL = "attacktype1"

NUMERIC_COLS = [
    "iyear",
    "imonth",
    "iday",
    "country",
    "region_code",
]

CAT_NOM_COLS = [
    "provstate",
    "city",
]

CAT_ORD_COLS = [
]

# Define Arguments for this step

class MyArgs:
    def __init__(self, /, **kwargs):
        self.__dict__.update(kwargs)

args = MyArgs(
                train_data = "/tmp/prep/train",
                model_output = "/tmp/train",
                regressor__n_estimators = 500,
                regressor__bootstrap = 1,
                regressor__max_depth = 10,
                regressor__max_features = "auto", 
                regressor__min_samples_leaf = 4,
                regressor__min_samples_split = 5
                )

os.makedirs(args.model_output, exist_ok = True)

# Set pyTorch local env to use segmented GPU memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Clear GPU cache & Set the device to use GPU
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------ LSTM Prediction Model ------------------------------

def train_model(X_tensor, Y_tensor, num_classes, sequence_length=30, hidden_size=128, num_epochs=10, batch_size=32):
    class LSTMPredictor(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMPredictor, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            lstm_out = self.dropout(lstm_out)
            logits = self.fc(lstm_out[:, -1, :])
            return logits

    # Create sequences
    def create_sequences(input_data, seq_length):
        sequences = []
        for i in range(len(input_data) - seq_length + 1):
            seq = input_data[i:i + seq_length]
            sequences.append(seq)
        return torch.stack(sequences)

    sequences = create_sequences(X_tensor, sequence_length)

    # Create DataLoader
    dataset = TensorDataset(sequences, Y_tensor[:len(sequences)])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = LSTMPredictor(input_size=X_tensor.shape[1], hidden_size=hidden_size, output_size=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Training loop
    for epoch in range(num_epochs):
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            outputs = model(batch_x)
            if batch_y.ndim > 1:
                batch_y = batch_y.argmax(dim=1)
            batch_y = batch_y.long()

            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    return model

# ------------------------------ Date Prediction Model ------------------------------

class LSTMDate(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMDate, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

# Convert date to numeric since the final day in dataset
def convert_date_to_numeric(date):
    return (date - datetime(2017, 12, 31)).days

# Generate date range (years, months, days)
def generate_date_range(start_year, end_year):
    date_list = []
    for year in range(start_year, end_year + 1):
      # Loop through months 1 to 12
        for month in range(1, 13):
            num_days = calendar.monthrange(year, month)[1]
            # Loop through the days of the month
            for day in range(1, num_days + 1):
                date = datetime(year, month, day)
                date_list.append(date)
    return date_list

# Create sequences from the numeric dates and features
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

# Main function to train the model
def train_date(X_tensor, Y_tensor, sequence_length=30, hidden_size=128, num_epochs=1000, batch_size=32):
    X_tensor = X_tensor.to(device)
    Y_tensor = Y_tensor.to(device)

    # Reshape X_tensor to have shape [num_samples, sequence_length, num_features]
    X_tensor = X_tensor.reshape(X_tensor.shape[0], X_tensor.shape[1], 1)

    # Create a DataLoader for batching
    dataset = TensorDataset(X_tensor, Y_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create the model
    input_size = X_tensor.shape[2]  # Number of features (year, month, day)
    output_size = 1  # Predicting a single value (days since reference date)
    model = LSTMDate(input_size, hidden_size, output_size)

    # Move model to device
    model.to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        for i, (inputs, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Zero the gradients, backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (i + 1) % 100 == 0:  # Print every 100 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}], Loss: {loss.item():.4f}")

    return model

# Generate date list for the years 2018 to 2023
date_list = generate_date_range(2018, 2023)

# Convert the list of dates to numeric values (days since 2017-12-31)
date_numeric = [convert_date_to_numeric(date) for date in date_list]

# Create sequences for training
# Use the last 30 days to predict the next one
sequence_length = 30
X, Y = create_sequences(date_numeric, sequence_length)

# Convert to PyTorch tensors
X_tensor_date = torch.tensor(X, dtype=torch.float32)
Y_tensor_date = torch.tensor(Y, dtype=torch.float32)

# ------------------------------ Train & Evaluate Models ------------------------------
model_attack = train_model(X_tensor, Y_tensor_attack, num_classes=num_attack_types)
model_attack = model_attack.to(device)

model_groups = train_model(X_tensor, Y_tensor_group, num_classes=num_groups)
model_groups = model_groups.to(device)

model_city = train_model(X_tensor, Y_tensor_city, num_classes=num_cities)
model_city = model_city.to(device)

model_provstate = train_model(X_tensor, Y_tensor_provstate, num_classes=num_provstates)
model_provstate = model_provstate.to(device)

model_date = train_date(X_tensor_date, Y_tensor_date, sequence_length=sequence_length)
model_date = model_date.to(device)

# Set the models to evaluation mode
model_attack.eval()
model_groups.eval()
model_city.eval()
model_provstate.eval()
model_date.eval()



def main(args):
    '''Read train dataset, train model, save trained model'''

    # Read train data
    train_data = pd.read_parquet(Path(args.train_data))

    # Split the data into input(X) and output(y)
    y_train = train_data[TARGET_COL]
    X_train = train_data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS]

    # Train a Random Forest Regression Model with the training set
    model = RandomForestRegressor(n_estimators = args.regressor__n_estimators,
                                  bootstrap = args.regressor__bootstrap,
                                  max_depth = args.regressor__max_depth,
                                  max_features = args.regressor__max_features,
                                  min_samples_leaf = args.regressor__min_samples_leaf,
                                  min_samples_split = args.regressor__min_samples_split,
                                  random_state=0)

    # log model hyperparameters
    mlflow.log_param("model", "RandomForestRegressor")
    mlflow.log_param("n_estimators", args.regressor__n_estimators)
    mlflow.log_param("bootstrap", args.regressor__bootstrap)
    mlflow.log_param("max_depth", args.regressor__max_depth)
    mlflow.log_param("max_features", args.regressor__max_features)
    mlflow.log_param("min_samples_leaf", args.regressor__min_samples_leaf)
    mlflow.log_param("min_samples_split", args.regressor__min_samples_split)

    # Train model with the train set
    model.fit(X_train, y_train)

    # Predict using the Regression Model
    yhat_train = model.predict(X_train)

    # Evaluate Regression performance with the train set
    r2 = r2_score(y_train, yhat_train)
    mse = mean_squared_error(y_train, yhat_train)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, yhat_train)
    
    # log model performance metrics
    mlflow.log_metric("train r2", r2)
    mlflow.log_metric("train mse", mse)
    mlflow.log_metric("train rmse", rmse)
    mlflow.log_metric("train mae", mae)

    # Visualize results
    plt.scatter(y_train, yhat_train,  color='black')
    plt.plot(y_train, y_train, color='blue', linewidth=3)
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.savefig("regression_results.png")
    mlflow.log_artifact("regression_results.png")

    # Save the model
    mlflow.sklearn.save_model(sk_model=model, path=args.model_output)

if __name__ == "__main__":
    mlflow.start_run()
    
    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()
    

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Model output path: {args.model_output}",
        f"n_estimators: {args.regressor__n_estimators}",
        f"bootstrap: {args.regressor__bootstrap}",
        f"max_depth: {args.regressor__max_depth}",
        f"max_features: {args.regressor__max_features}",
        f"min_samples_leaf: {args.regressor__min_samples_leaf}",
        f"min_samples_split: {args.regressor__min_samples_split}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
