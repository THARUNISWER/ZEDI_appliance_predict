from keras.models import model_from_json, Model
from keras.layers import Dense, LSTM, Dropout, Input
from numpy import load
from keras import backend as K
import random
import sys

# inputs
WINDOW_SIZE = 800
TRAINING_BATCH_SIZE = 128  # model metric
EPOCHS = 1
reset = 1  # 1 for resetting and 0 to continue from where it left off
model_weight_output_json = "./fridge_model_data/fri_bin_test.json"
model_weight_output_h5 = "./fridge_model_data/fri_bin_test.h5"
geyser_model_json = "../Geyser/geyser_model_data/gey_sig_test.json"
geyser_model_h5 = "../Geyser/geyser_model_data/gey_sig_test.h5"
input_data_a = 'D:\\fridge_data_a_test.npy'
input_data_b = 'D:\\fridge_data_b_test.npy'

if reset == 1:
    res = int(input("Are you sure you want to reset? (0/1)"))
    if res == 0:
        sys.exit()
    print("System reset for fridge")
else:
    print("Continuing where its left off for fridge")

# loading data
data_a = load(input_data_a, mmap_mode="r")
data_b = load(input_data_b, mmap_mode="r")

# loading geyser model
json_file = open(geyser_model_json, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_gey = model_from_json(loaded_model_json)
loaded_model_gey.load_weights(geyser_model_h5)
print("Loaded geyser model from disk")

loaded_model_gey.compile(loss='mean_squared_error', optimizer='adam')


# Defining custom accuracy metric function
def accuracy(y_true, y_pred):
    acc = K.mean(K.equal(K.round(y_true[0]), K.round(y_pred[0])), axis=-1)
    return acc


# Building model
if reset == 1:
    input1 = Input(shape=(WINDOW_SIZE, 1))
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
    output = Dense(units=WINDOW_SIZE, activation='sigmoid')(x)
    fridge_model = Model(inputs=[input1], outputs=output)

    fridge_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[accuracy])
else:
    json_file = open(model_weight_output_json, 'r')
    loaded_fridge_json = json_file.read()
    json_file.close()
    fridge_model = model_from_json(loaded_fridge_json)
    fridge_model.load_weights(model_weight_output_h5)
    print("Loaded fridge model from disk")

    fridge_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[accuracy])


print("Fridge model ready")
print(fridge_model.summary())

# if geyser is not capable of window size

# mod_data_b = np.empty([0, 100], dtype=float)
# for d in data_b:
#     for i in range(MAX_NUM_ROWS//100):
#         mod_data_b = np.append(mod_data_b, d[i*100:i*100 + 100].reshape(1, 100), axis=0)
#
# temp_data = loaded_model_gey.predict(mod_data_b)
# temp = np.empty(0)
# data_c = np.empty([0, MAX_NUM_ROWS], dtype=float)
# for d in temp_data:
#     temp = np.concatenate((temp, d))
#     if temp.size == MAX_NUM_ROWS:
#         data_c = np.append(data_c, temp.reshape(1, MAX_NUM_ROWS), axis=0)
#         temp = np.empty(0)

print("Geyser Prediction in process..")
data_c = loaded_model_gey.predict(data_b)

for k in range(EPOCHS):
    print("Model training in progress for epoch " + str(k))
    fridge_model.fit([data_b - data_c], data_a, validation_split=0.1, epochs=1, batch_size=TRAINING_BATCH_SIZE, shuffle=True)
    print("Model training completed :)")

    model_json = fridge_model.to_json()
    with open(model_weight_output_json, "w") as json_file:
        json_file.write(model_json)
    fridge_model.save_weights(model_weight_output_h5)
    print("Saved model to disk")

# random test
print("Random testing for a single data point")
ind = random.randint(0, len(data_a) - 1)
data_c = loaded_model_gey.predict(data_b[ind: ind + 1])
data_d = fridge_model.predict([data_b[ind: ind + 1] - data_c])
print("Input: " + str(data_b[ind: ind + 1]))
print("Actual output: " + str(data_a[ind: ind + 1]))
print("Predicted output: " + str(data_d))
