from keras.models import model_from_json, Model
from keras.layers import Dense, LSTM, Dropout, Input, concatenate
import pandas as pd
import numpy as np
from datetime import datetime

'''
Both fridge and washing machine as output
'''

WINDOW_SIZE = 100
MAX_NUM_ROWS = WINDOW_SIZE
FORMAT = "%d/%m/%Y %H:%M:%S"
FRI_THRESHOLD = 0.2
WMG_THRESHOLD = 0.1
LEG_BATCH_THRESHOLD = WINDOW_SIZE

target_file_fridge1 = "D:\\RNN_Data\\Data\\FG Split\\FG_till24.CSV"
target_file_fridge2 = "D:\\RNN_Data\\Data\\FG Split\\FG_from25.CSV"
target_file_wmg = "D:\\RNN_Data\\Data\\WMG\\DATALOG.CSV"

input_main_file = "D:\\preprocessed_data\\train_main_file.csv"


fridge_df1 = pd.read_csv(target_file_fridge1)
fridge_df1.columns = ['Time', 'Temperature', 'Humidity', 'Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power Factor']
fridge_df2 = pd.read_csv(target_file_fridge2)
fridge_df2.columns = ['Time', 'Temperature', 'Humidity', 'Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power Factor']
fridge_df = pd.concat([fridge_df1, fridge_df2], ignore_index=True)
# removing nan values
fridge_df.dropna()
fridge_df.drop(fridge_df[fridge_df["Current"] == "  NAN "].index, inplace=True)
fridge_df.drop(fridge_df[fridge_df["Frequency"] == ""].index, inplace=True)
fridge_df = fridge_df[["Time", "Current", "Voltage", "Power", "Power Factor"]]

wmg_df = pd.read_csv(target_file_wmg)
wmg_df.columns = ['Time', 'Temperature', 'Humidity', 'Voltage', 'Current', 'Power', 'Energy', 'Frequency', 'Power Factor']
# removing nan values
wmg_df.dropna()
wmg_df.drop(wmg_df[wmg_df["Current"] == "  NAN "].index, inplace=True)
wmg_df.drop(wmg_df[wmg_df["Frequency"] == ""].index, inplace=True)
wmg_df = wmg_df[["Time", "Current", "Voltage", "Power", "Power Factor"]]

main_df = pd.read_csv(input_main_file)

# load json and create model
json_file = open('../model_data/geyser_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_gey = model_from_json(loaded_model_json)
# load weights into new model
loaded_model_gey.load_weights("../model_data/geyser_model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model_gey.compile(loss='mean_squared_error', optimizer='adam')

input1 = Input(shape=(WINDOW_SIZE, 1))
# input2 = Input(shape=(100, 1))
# x = concatenate([input1, input2])
x = input1
x = LSTM(units=128, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(units=128, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(units=128, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(units=128, return_sequences=True)(x)
x = Dropout(0.2)(x)
x = LSTM(units=128)(x)
output1 = Dense(units=WINDOW_SIZE, activation='sigmoid')(x)
output2 = Dense(units=WINDOW_SIZE, activation='sigmoid')(x)
model = Model(inputs=[input1], outputs=[output1, output2])

model.compile(optimizer='adam', loss='mean_squared_error')

i = 100
j = 0
k = 100
main_data = []
fri_label = []
wmg_label = []
data_a = np.empty([0, MAX_NUM_ROWS], dtype=float)
data_b = np.empty([0, MAX_NUM_ROWS], dtype=float)
data_c = np.empty([0, MAX_NUM_ROWS], dtype=float)
batch_ind = 0

print("For loop started..")
while j < len(main_df) and i < len(fridge_df) and k < len(wmg_df):
    # print(i)
    try:
        main_dt = datetime.strptime(main_df.loc[j, "Start_time"], FORMAT)
        fri_dt = datetime.strptime(str(fridge_df.loc[i, "Time"]), FORMAT)
        wmg_dt = datetime.strptime(str(wmg_df.loc[k, "Time"]), FORMAT)

        # adjuster
        while fri_dt < main_dt and i < len(fridge_df):
            i = i + 1
            fri_dt = datetime.strptime(str(fridge_df.loc[i, "Time"]), FORMAT)

        # adjuster
        while wmg_dt < main_dt and k < len(wmg_df):
            k = k + 1
            wmg_dt = datetime.strptime(str(wmg_df.loc[k, "Time"]), FORMAT)

        temp_j = j
        temp_i = i
        temp_k = k
        while main_dt == fri_dt == wmg_dt:
            main_data.append(float(main_df.loc[temp_j, "Current"]))
            fri_label.append(1 if float(fridge_df.loc[temp_i, "Current"]) > FRI_THRESHOLD else 0)
            wmg_label.append(1 if float(wmg_df.loc[temp_k, "Current"]) > WMG_THRESHOLD else 0)
            temp_i += 1
            temp_j += 1
            temp_k += 1
            main_dt = datetime.strptime(main_df.loc[temp_j, "Start_time"], FORMAT)
            fri_dt = datetime.strptime(fridge_df.loc[temp_i, "Time"], FORMAT)
            wmg_dt = datetime.strptime(str(wmg_df.loc[temp_k, "Time"]), FORMAT)
            if len(fri_label) == WINDOW_SIZE:
                X1 = np.array(fri_label)
                X1 = X1.reshape(1, len(X1))
                len_to_pad = MAX_NUM_ROWS - X1[0].size
                X1 = np.pad(X1, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))

                X2 = np.array(main_data)
                X2 = X2.reshape(1, len(X2))
                len_to_pad = MAX_NUM_ROWS - X2[0].size
                X2 = np.pad(X2, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))

                X3 = np.array(wmg_label)
                X3 = X3.reshape(1, len(X3))
                len_to_pad = MAX_NUM_ROWS - X3[0].size
                X3 = np.pad(X3, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))

                data_a = np.append(data_a, X1, axis=0)
                data_b = np.append(data_b, X2, axis=0)
                data_c = np.append(data_c, X3, axis=0)

                # print(main_dt)
                # print(main_data)
                # print(fri_label)
                # print(wmg_label)
                print(batch_ind)
                batch_ind += 1
                main_data = []
                fri_label = []
                wmg_label = []
                i = i + 1
                k = k + 1
                break

            elif main_dt != fri_dt:
                main_data = []
                fri_label = []
                wmg_label = []
                i = i + 1
                k = k + 1

    except Exception as e:
        print(e)
        main_data = []
        fri_label = []
        wmg_label = []
        print("Error in i:" + str(i) + " j:" + str(j) + "k: " + str(k))
        i += 1
        k += 1
    if i == 10000:
        break
    j += 1

print("Model training in progress..")
data_d = loaded_model_gey.predict(data_b)

model.fit([data_b - data_d], [data_a, data_c], epochs=10, batch_size=128, shuffle=True)
out = model.predict([data_b[1:1000] - data_d[1:1000]])

for i in range(len(out)):
    print("Input: " + str(data_b[i] - data_d[i]))
    print("Expected output fridge: " + str(data_a[i]))
    print("Output fridge: " + str(out[0][i]))
    print("Expected output wmg: " + str(data_c[i]))
    print("Output wmg: " + str(out[1][i]))
print("Model training completed :)")

# serialize model to JSON
model_json = model.to_json()
with open("../model_data/fridge_pos1_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../model_data/fridge_pos1_model.h5")
print("Saved model to disk")