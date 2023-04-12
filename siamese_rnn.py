from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dot, Lambda, Concatenate
import keras.backend as K
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Lambda
from keras.optimizers import Adam
import pandas as pd
from io import StringIO
import ast
import sys

train_file_geyser = "D:\\RNN_Data\\Data\\GG\\DATALOG.CSV"
train_df_geyser = pd.read_csv(train_file_geyser)
train_df_geyser = train_df_geyser[["Data_a", "Data_b", "Label"]]
# for ind in train_df_geyser.index:
#     df_a = pd.DataFrame.from_dict(ast.literal_eval(train_df_geyser["Data_a"][ind]))
#     print(np.array(data_a["Current"], dtype='float64'))
#     df_b = pd.DataFrame.from_dict(ast.literal_eval(train_df_geyser["Data_b"][ind]))
#     print(np.array(data_b["Current"], dtype='float64'))
from keras.layers import Input, LSTM, Dense, Concatenate, Masking
from keras.models import Model
# Define the maximum input shape
max_num_rows = 500
num_cols = 1
max_input_shape = (max_num_rows, num_cols)

# Define the RNN architecture
hidden_size = 10
rnn = LSTM(units=hidden_size, activation='tanh')

# Define the input layers for the two dataframes
input1 = Input(shape=max_input_shape)
input2 = Input(shape=max_input_shape)

# Define the masking layers for the two input layers
mask1 = Masking(mask_value=0.0)(input1)
mask2 = Masking(mask_value=0.0)(input2)

# Define the RNN layers for the two input layers
rnn1 = rnn(mask1)
rnn2 = rnn(mask2)

# Define the merge layer to concatenate the outputs of the two RNN layers
merged = Concatenate()([rnn1, rnn2])

# Define the fully connected layer to predict the similarity score
dense = Dense(units=1, activation='sigmoid')(merged)

# Define the Siamese RNN model
model = Model(inputs=[input1, input2], outputs=dense)

# Compile the model with binary cross-entropy loss and an optimizer
model.compile(loss='binary_crossentropy', optimizer='adam')

# train the model
print("Training...")
data_a = np.empty([0, 500], dtype= float)
data_b = np.empty([0, 500], dtype=float)
labels = np.empty(0, dtype=int)
convert_dict = {'Current': float}
for ind in range(0, 100):
    df_a = pd.DataFrame.from_dict(ast.literal_eval(train_df_geyser["Data_a"][ind]))
    # df_a.astype(convert_dict)
    X1 = np.array(df_a["Current"], dtype='float64')
    X1 = X1.reshape(1, len(X1))
    len_to_pad = max_num_rows - X1[0].size
    X1 = np.pad(X1, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))
    df_b = pd.DataFrame.from_dict(ast.literal_eval(train_df_geyser["Data_b"][ind]))
    # df_b.astype(convert_dict)
    X2 = np.array(df_b["Current"], dtype='float64')
    X2 = X2.reshape(1, len(X2))
    len_to_pad = max_num_rows - X2[0].size
    X2 = np.pad(X2, ((0,0), (0, len_to_pad)), 'constant', constant_values=(0,0))
    y = np.array(train_df_geyser["Label"][ind]).reshape(1)
    # print(X1)
    # print(X2)
    # print(X2[0].size)
    # print(y)
    data_a = np.append(data_a, X1, axis=0)
    data_b = np.append(data_b, X2, axis=0)
    labels = np.append(labels, y)
    print(ind)
print(data_a)
print(data_b)
print(labels)
model.fit([data_a, data_b], labels, epochs=100, batch_size=32)
print("Completed Training :)")

data_a = np.empty([0, 500], dtype= float)
data_b = np.empty([0, 500], dtype=float)
labels = np.empty(0, dtype=int)
for ind in range(100,152):
    df_a = pd.DataFrame.from_dict(ast.literal_eval(train_df_geyser["Data_a"][ind]))
    # df_a.astype(convert_dict)
    X1 = np.array(df_a["Current"], dtype='float64')
    X1 = X1.reshape(1, len(X1))
    len_to_pad = max_num_rows - X1[0].size
    X1 = np.pad(X1, ((0, 0), (0, len_to_pad)), 'constant', constant_values=(0, 0))
    df_b = pd.DataFrame.from_dict(ast.literal_eval(train_df_geyser["Data_b"][ind]))
    # df_b.astype(convert_dict)
    X2 = np.array(df_b["Current"], dtype='float64')
    X2 = X2.reshape(1, len(X2))
    len_to_pad = max_num_rows - X2[0].size
    X2 = np.pad(X2, ((0,0), (0, len_to_pad)), 'constant', constant_values=(0,0))
    y = np.array(train_df_geyser["Label"][ind]).reshape(1)
    # print(X1)
    # print(X2)
    # print(X2[0].size)
    # print(y)
    data_a = np.append(data_a, X1, axis=0)
    data_b = np.append(data_b, X2, axis=0)
    labels = np.append(labels, y)
    print(ind)
# evaluate the model
arr = model.predict([data_a, data_b])
print(arr)
scores = model.evaluate([data_a, data_b], labels, verbose=0)
print(model.metrics_names)
print(scores)
# sys.exit()
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# serialize model to JSON
model_json = model.to_json()
with open("model_data/geyser_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_data/geyser_model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model_data/geyser_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_data/geyser_model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate([data_a, data_b], labels, verbose=0)
print(score)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))