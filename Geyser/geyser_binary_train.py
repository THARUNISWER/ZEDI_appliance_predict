from keras.models import Sequential, model_from_json
from keras.layers import Dense, LSTM, Dropout
from keras import backend as K
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import os
import time
import random
import sys

'''
A model which predicts 1 if geyser was on, otherwise 0
'''

# inputs
WINDOW_SIZE = 1000
TRAINING_BATCH_SIZE = 128  # model metric
TRAIN_BATCHES = 200  # for max efficiency give it in multiples of TRAINING_BATCH_SIZE
EPOCHS = 1
SLEEP_TIMEOUT = 10  # sleeping time between TRAIN_BATCHES
reset = 1  # 1 for resetting and 0 to continue from where it left off
target_file_geyser = "D:\\RNN_Data\\Data\\GG\\DATALOG.CSV"
input_main_file = "D:\\preprocessed_data\\train_main_file.csv"
model_weight_output_json = "./geyser_model_data/gey_bin_1000.json"
model_weight_output_h5 = "./geyser_model_data/gey_bin_1000.h5"
store_variables = "./geyser_objects/gey_bin.pkl"


if reset == 1:
    res = int(input("Are you sure you want to reset? (0/1)"))
    if res == 0:
        sys.exit()
    print("System reset for geyser")
else:
    print("Continuing where its left off for geyser")

# constants
FORMAT = "%d/%m/%Y %H:%M:%S"
GEY_THRESHOLD = 5.0

# data preprocessing
geyser_df = pd.read_csv(target_file_geyser)
geyser_df.columns = ['Time', 'Temperature', 'Humidity', 'Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power Factor']
geyser_df.dropna()
geyser_df.drop(geyser_df[geyser_df["Current"] == "  NAN "].index, inplace=True)
geyser_df.drop(geyser_df[geyser_df["Frequency"] == ""].index, inplace=True)
geyser_df = geyser_df[["Time", "Current", "Voltage", "Power", "Power Factor"]]
main_df = pd.read_csv(input_main_file)


# Defining custom accuracy metric function
def accuracy(y_true, y_pred):
    acc = K.mean(K.equal(K.round(y_true[0]), K.round(y_pred[0])), axis=-1)
    return acc


# Building model
if reset == 1:
    model = Sequential()
    model.add(LSTM(units=128, input_shape=(WINDOW_SIZE, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128))
    model.add(Dense(units=WINDOW_SIZE, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[accuracy])
else:
    json_file = open(model_weight_output_json, 'r')
    loaded_geyser_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_geyser_json)
    model.load_weights(model_weight_output_h5)
    print("Loaded model from disk")

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# processing data
i = 100
j = 0
batch_ind = 0
batch_0_ind = 0
batch_1_ind = 0

if os.path.exists(store_variables) and reset == 0:
    with open(store_variables, 'rb') as f:
        i, j, batch_ind, batch_0_ind, batch_1_ind = pickle.load(f)

print("Starting from i: " + str(i) + " j: " + str(j) + " batch_ind: " + str(batch_ind))

main_data = []
labels = []
data_a = np.empty([0, WINDOW_SIZE], dtype=float)
data_b = np.empty([0, WINDOW_SIZE], dtype=float)


# function to train model
def gey_train():
    # training
    for k in range(EPOCHS):
        print("Model training in progress for epoch " + str(k))
        model.fit(data_b, data_a, validation_split=0.2, epochs=1, batch_size=TRAINING_BATCH_SIZE, shuffle=True)
        print("Model training completed :)")

        model_json = model.to_json()
        with open(model_weight_output_json, "w") as json_file:
            json_file.write(model_json)
        model.save_weights(model_weight_output_h5)
        print("Saved model to disk")

    # random test
    print("Random testing for a single data point")
    ind = random.randint(0, TRAIN_BATCHES - 1)
    data_c = model.predict(data_b[ind: ind + 1])
    print("Input: " + str(data_b[ind: ind + 1]))
    print("Actual output: " + str(data_a[ind: ind + 1]))
    print("Predicted output: " + str(data_c))

print("For loop started..")
while j < len(main_df) and i < len(geyser_df):
    try:
        main_dt = datetime.strptime(main_df.loc[j, "Start_time"], FORMAT)
        gey_dt = datetime.strptime(geyser_df.loc[i, "Time"], FORMAT)

        # adjuster
        while gey_dt < main_dt and i < len(geyser_df):
            i = i + 1
            gey_dt = datetime.strptime(geyser_df.loc[i, "Time"], FORMAT)

        if main_dt == gey_dt:
            temp_j = j
            temp_i = i
            while True:
                main_data.append(float(main_df.loc[temp_j, "Current"]))
                labels.append(1 if float(geyser_df.loc[temp_i, "Current"]) > GEY_THRESHOLD else 0)
                temp_i += 1
                temp_j += 1
                main_dt = datetime.strptime(main_df.loc[temp_j, "Start_time"], FORMAT)
                gey_dt = datetime.strptime(geyser_df.loc[temp_i, "Time"], FORMAT)

                if main_dt < gey_dt:
                    temp_j += 1
                elif gey_dt < main_dt:
                    temp_i += 1

                if len(labels) == WINDOW_SIZE:
                    X1 = np.array(labels)
                    X1 = X1.reshape(1, len(X1))
                    len_to_pad = WINDOW_SIZE - X1[0].size
                    X1 = np.pad(X1, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))

                    X2 = np.array(main_data)
                    X2 = X2.reshape(1, len(X2))
                    len_to_pad = WINDOW_SIZE - X2[0].size
                    X2 = np.pad(X2, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))

                    data_a = np.append(data_a, X1, axis=0)
                    data_b = np.append(data_b, X2, axis=0)

                    if X1[0][0] == 1:
                        batch_1_ind += 1
                    else:
                        batch_0_ind += 1

                    batch_ind += 1
                    if batch_ind % 100 == 0:
                        print(str(batch_ind) + " batches over")
                    if batch_ind % TRAIN_BATCHES == 0:
                        # Saving the variables:
                        with open(store_variables, 'wb') as f:
                            pickle.dump([i, j, batch_ind, batch_0_ind, batch_1_ind], f)
                        print("No. of batches(windows) processed: " + str(batch_ind))
                        print("No. of batches(windows) in which geyser was on: " + str(batch_1_ind))
                        print("No. of batches(windows) in which geyser was off: " + str(batch_0_ind))
                        gey_train()
                        print("To exit stop the program")
                        time.sleep(SLEEP_TIMEOUT)
                        print("Continuing....")
                    main_data = []
                    labels = []
                    i = i + 1
                    break

    except Exception as e:
        print(e)
        main_data = []
        labels = []
        print("Error in i:" + str(i) + " j:" + str(j))
        i += 1
    j += 1

