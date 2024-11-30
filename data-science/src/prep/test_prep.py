import argparse

from pathlib import Path
import os
import numpy as np
import pandas as pd

import mlflow

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

# ------------------------------ Data Preprocessing ------------------------------

# Set pyTorch local env to use segmented GPU memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Clear GPU cache & Set the device to use GPU
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
# Skip rows = 1 because those are the column names
X = np.array([])

# Read the file using its encoding
data = pd.read_csv('./globalterrorismdb_0718dist.csv', encoding="Windows-1252")

# Extract relevant columns (adjust indices or column names as needed)
input_columns = data.iloc[:, [1, 2, 3, 7, 11]]
input_columns = input_columns.fillna(0)

# Convert non-numeric to numeric and fill missing values
for col in input_columns.columns:
    input_columns[col] = pd.to_numeric(input_columns[col], errors='coerce')  # Convert non-numeric to NaN
input_columns = input_columns.fillna(0)  # Replace NaN with 0

attack_target = data.iloc[:, [28]]
group_target = data.iloc[:, [58]]

# Set the base date (last day of 2017)
last_date = datetime(2017, 12, 31)

# Convert last date to numeric form
last_date_numeric = last_date.toordinal()

# Get date from dataset
data['date_str'] = data['iyear'].astype(str) + '-' + data['imonth'].astype(str).str.zfill(2) + '-' + data['iday'].astype(str).str.zfill(2)
data['date'] = pd.to_datetime(data['date_str'], errors='coerce')


# Convert dates to numeric by subtracting the last date of 2017
# Get number of days since Dec 31, 2017
data['date_numeric'] = (data['date'] - last_date).dt.days

# Extract unique values
unique_attacks = list(set(data['attacktype1_txt']))
unique_groups = list(set(data['gname']))
unique_provstates = list(set(data['provstate']))
unique_cities = list(set(data['city']))

# Initialize LabelEncoder and fit to the unique groups
attack_encoder = LabelEncoder()
attack_encoder.fit(unique_attacks)

group_encoder = LabelEncoder()
group_encoder.fit(unique_groups)

provstate_encoder = LabelEncoder()
provstate_encoder.fit(unique_provstates)

city_encoder = LabelEncoder()
city_encoder.fit(unique_cities)

# Set the output size based on the number of unique attack types
num_attack_types = len(unique_attacks)
num_groups = len(unique_groups)
num_cities = len(unique_cities)
num_provstates = len(unique_provstates)

# Create a dictionary to map names to their encoded IDs
group_dict = pd.Series(group_encoder.transform(unique_groups), index=unique_groups)
provstate_dict = pd.Series(provstate_encoder.transform(unique_provstates), index=unique_provstates)
city_dict = pd.Series(city_encoder.transform(unique_cities), index=unique_cities)

# Assign values to tensors
input_tensor = torch.tensor(input_columns.to_numpy(), dtype=torch.float32)
attack_target_tensor = torch.tensor(attack_target.values, dtype=torch.float32)
group_target_tensor = torch.tensor(group_encoder.fit_transform(group_target.values), dtype=torch.float32)
city_target_tensor = torch.tensor(city_encoder.fit_transform(data['city'].values), dtype=torch.float32)
provstate_target_tensor = torch.tensor(provstate_encoder.fit_transform(data['provstate'].values), dtype=torch.float32)

# TESTING - PRINT DICTIONARY ITEMS
#for key, value in group_dict.items():
#  print("group: ", key, "| ID #:", value)

#for key, value in provstate_dict.items():
#  print("provstate: ", key, "| ID #:", value)

#for key, value in city_dict.items():
#  print("city: ", key, "| ID #:", value)

# Assign values to tensors for processing
X_tensor = input_tensor

# Normalize: mean and std for each feature
mean = X_tensor.mean(dim=0, keepdim=True)
std = X_tensor.std(dim=0, keepdim=True)
X_tensor = (X_tensor - mean) / std

Y_tensor_attack = attack_target_tensor
Y_tensor_group = group_target_tensor
Y_tensor_city = city_target_tensor
Y_tensor_provstate = provstate_target_tensor
Y_tensor_date = torch.tensor(data['date_numeric'] - last_date_numeric, dtype=torch.float32)

# Set tensors to use GPU
X_tensor = X_tensor.to(device)
Y_tensor_attack = Y_tensor_attack.to(device)
Y_tensor_group = Y_tensor_group.to(device)
Y_tensor_city = Y_tensor_city.to(device)
Y_tensor_provstate = Y_tensor_provstate.to(device)
Y_tensor_date = Y_tensor_date.to(device)

# Define Arguments for this step

class MyArgs:
    def __init__(self, /, **kwargs):
        self.__dict__.update(kwargs)

args = MyArgs(
            raw_data = "../../data/", 
            train_data = "/tmp/prep/train",
            val_data = "/tmp/prep/val",
            test_data = "/tmp/prep/test",
            )

os.makedirs(args.train_data, exist_ok = True)
os.makedirs(args.val_data, exist_ok = True)
os.makedirs(args.test_data, exist_ok = True)



def main(args):
    '''Read, split, and save datasets'''

    # ------------ Reading Data ------------ #
    # -------------------------------------- #

    print("mounted_path files: ")
    arr = os.listdir(args.raw_data)
    print(arr)

    data = pd.read_csv((Path(args.raw_data) / 'globalterrorismdb_0718dist.csv'))
    data = data[NUMERIC_COLS + CAT_NOM_COLS + CAT_ORD_COLS + [TARGET_COL]]

    # ------------- Split Data ------------- #
    # -------------------------------------- #

    # Split data into train, val and test datasets

    random_data = np.random.rand(len(data))

    msk_train = random_data < 0.7
    msk_val = (random_data >= 0.7) & (random_data < 0.85)
    msk_test = random_data >= 0.85

    train = data[msk_train]
    val = data[msk_val]
    test = data[msk_test]

    mlflow.log_metric('train size', train.shape[0])
    mlflow.log_metric('val size', val.shape[0])
    mlflow.log_metric('test size', test.shape[0])

    train.to_parquet((Path(args.train_data) / "train.parquet"))
    val.to_parquet((Path(args.val_data) / "val.parquet"))
    test.to_parquet((Path(args.test_data) / "test.parquet"))

if __name__ == "__main__":

    mlflow.start_run()
    
    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Raw data path: {args.raw_data}",
        f"Train dataset output path: {args.train_data}",
        f"Val dataset output path: {args.val_data}",
        f"Test dataset path: {args.test_data}",

    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
