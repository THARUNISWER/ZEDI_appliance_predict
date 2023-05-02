import pandas as pd
import numpy as np
import sys
from datetime import datetime, timedelta
import pickle
import os
import time
from npy_append_array import NpyAppendArray, recover

'''
fridge data processor with variable window size
'''

# inputs
WINDOW_SIZE = 800
TRAINING_BATCH_SIZE = 128
TRAIN_BATCHES = 128*3  # for max efficiency give it in multiples of TRAINING_BATCH_SIZE(128 or 256)
SLEEP_TIMEOUT = 10  # sleeping time between data batches
reset = 0  # 1 for resetting and 0 to continue from where it left off
target_file_fridge1 = "D:\\RNN_Data\\Data\\FG Split\\FG_till24.csv"
target_file_fridge2 = "D:\\RNN_Data\\Data\\FG Split\\FG_from25.csv"
input_main_file = "D:\\preprocessed_data\\train_main_file.csv"
output_data_a = 'D:\\fridge_data_a_test.npy'
output_data_b = 'D:\\fridge_data_b_test.npy'
store_variables = "./fridge_objects/fri_bin_test.pkl"

if reset == 1:
    res = int(input("Are you sure you want to reset? (0/1)"))
    if res == 0:
        sys.exit()
    print("System reset for fridge data collection")
else:
    print("Continuing where its left off for fridge")

# constants
FORMAT = "%d/%m/%Y %H:%M:%S"
FRI_THRESHOLD = 0.2

# data preprocessing
fridge_df1 = pd.read_csv(target_file_fridge1)
fridge_df1.columns = ['Time', 'Temperature', 'Humidity', 'Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power Factor']
fridge_df2 = pd.read_csv(target_file_fridge2)
fridge_df2.columns = ['Time', 'Temperature', 'Humidity', 'Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power Factor']
fridge_df = pd.concat([fridge_df1, fridge_df2], ignore_index=True)
fridge_df.dropna()
fridge_df.drop(fridge_df[fridge_df["Current"] == "  NAN "].index, inplace=True)
fridge_df.drop(fridge_df[fridge_df["Frequency"] == ""].index, inplace=True)
fridge_df = fridge_df[["Time", "Current", "Voltage", "Power", "Power Factor"]]
main_df = pd.read_csv(input_main_file)

if reset == 1:
    np_data_a = NpyAppendArray(output_data_a, delete_if_exists=True)
    np_data_b = NpyAppendArray(output_data_b, delete_if_exists=True)
else:
    try:
        np_data_a = NpyAppendArray(output_data_a)
        np_data_b = NpyAppendArray(output_data_b)
    except Exception as e:
        print(e)
        print("Recovery running...")
        recover(output_data_a)
        recover(output_data_b)
        print("Recovered succesfully :)")
        np_data_a = NpyAppendArray(output_data_a)
        np_data_b = NpyAppendArray(output_data_b)

# processing data
i = 1000
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

print("For loop started..")
while j < len(main_df) and i < len(fridge_df):
    try:
        main_dt = datetime.strptime(main_df.loc[j, "Start_time"], FORMAT)
        fri_dt = datetime.strptime(str(fridge_df.loc[i, "Time"]), FORMAT)

        if main_dt == fri_dt + timedelta(seconds=1):
            # print(main_dt)
            i += 1
            fri_dt = datetime.strptime(fridge_df.loc[i, "Time"], FORMAT)

        if main_dt == fri_dt - timedelta(seconds=1):
            i -= 1
            fri_dt = main_dt

        # adjuster
        while fri_dt < main_dt and i < len(fridge_df):
            i = i + 1
            fri_dt = datetime.strptime(str(fridge_df.loc[i, "Time"]), FORMAT)

        temp_j = j
        temp_i = i
        while main_dt == fri_dt:
            main_data.append(float(main_df.loc[temp_j, "Current"]))
            labels.append(1 if float(fridge_df.loc[temp_i, "Current"]) > FRI_THRESHOLD else 0)
            temp_i += 1
            temp_j += 1
            main_dt = datetime.strptime(main_df.loc[temp_j, "Start_time"], FORMAT)
            fri_dt = datetime.strptime(fridge_df.loc[temp_i, "Time"], FORMAT)

            if main_dt == fri_dt + timedelta(seconds=1):
                temp_i += 1
                fri_dt = datetime.strptime(fridge_df.loc[temp_i, "Time"], FORMAT)

            if main_dt == fri_dt - timedelta(seconds=1):
                temp_i -= 1
                fri_dt = main_dt

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
                if batch_ind % TRAINING_BATCH_SIZE == 0:
                    print(str(batch_ind) + " batches over")
                if batch_ind % TRAIN_BATCHES == 0:
                    print("No. of batches(windows) processed: " + str(batch_ind))
                    print("No. of batches(windows) in which fridge was on: " + str(batch_1_ind))
                    print("No. of batches(windows) in which fridge was off: " + str(batch_0_ind))
                    print("Saving data to disk")
                    np_data_a.append(data_a)
                    np_data_b.append(data_b)
                    print("Completed " + str(batch_ind) + " batches")
                    print("Current data file size of data_a is: " + str(os.path.getsize(output_data_a)) + " bytes")
                    np_data_a.close()
                    np_data_b.close()
                    # Saving the variables:
                    with open(store_variables, 'wb') as f:
                        pickle.dump([i, j, batch_ind, batch_0_ind, batch_1_ind], f)
                    print("To exit stop the program")
                    time.sleep(SLEEP_TIMEOUT)
                    np_data_a = NpyAppendArray(output_data_a)
                    np_data_b = NpyAppendArray(output_data_b)
                    data_a = np.empty([0, WINDOW_SIZE], dtype=float)
                    data_b = np.empty([0, WINDOW_SIZE], dtype=float)
                    print("Continuing....")
                main_data = []
                labels = []
                i = i + 1
                break

            elif main_dt != fri_dt:
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