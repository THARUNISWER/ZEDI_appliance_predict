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

train_file_geyser = "D:\\preprocessed_data\\geyser_train.csv"
train_df_geyser = pd.read_csv(train_file_geyser)
train_df_geyser = train_df_geyser[["Data_a", "Data_b", "Label"]]
# for ind in train_df_geyser.index:
#     df_a = pd.DataFrame.from_dict(ast.literal_eval(train_df_geyser["Data_a"][ind]))
#     print(np.array(data_a["Current"], dtype='float64'))
#     df_b = pd.DataFrame.from_dict(ast.literal_eval(train_df_geyser["Data_b"][ind]))
#     print(np.array(data_b["Current"], dtype='float64'))


max_len = 52
# Define the input layers
input_a = Input(shape=(max_len,))
input_b = Input(shape=(max_len,))

# Define the embedding layer
vocab_size = 9
embedding_dim = 32
embedding = Embedding(vocab_size, embedding_dim)

# Apply the embedding to each input
embedded_a = embedding(input_a)
embedded_b = embedding(input_b)

# Define the LSTM layer
hidden_size = 10
lstm = LSTM(hidden_size)

# Apply the LSTM to each input
encoded_a = lstm(embedded_a)
encoded_b = lstm(embedded_b)

# Compute a dot product between the encoded inputs
dot_product = Dot(axes=-1)([encoded_a, encoded_b])

# Compute the L2 norm of the encoded inputs
norm_a = Lambda(lambda x: K.l2_normalize(x, axis=-1))(encoded_a)
norm_b = Lambda(lambda x: K.l2_normalize(x, axis=-1))(encoded_b)

# Compute the cosine similarity between the encoded inputs
cosine_similarity = Dot(axes=-1)([norm_a, norm_b])

# Concatenate the dot product and cosine similarity
merged_vector = Concatenate()([dot_product, cosine_similarity])

# Add a dense layer for classification
predictions = Dense(1, activation='sigmoid')(merged_vector)

# Define the model
model = Model(inputs=[input_a, input_b], outputs=predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model

print("Training...")
convert_dict = {'Current': float}
for ind in train_df_geyser.index:
    df_a = pd.DataFrame.from_dict(ast.literal_eval(train_df_geyser["Data_a"][ind]))
    # df_a.astype(convert_dict)
    X1 = np.array(df_a["Current"], dtype='float64')
    X1 = X1.reshape(1, len(X1))
    df_b = pd.DataFrame.from_dict(ast.literal_eval(train_df_geyser["Data_b"][ind]))
    # df_b.astype(convert_dict)
    X2 = np.array(df_b["Current"], dtype='float64')
    X2 = X2.reshape(1, len(X2))
    y = np.array(train_df_geyser["Label"][ind]).reshape(1)
    print(X1)
    print(X2)
    print(y)
    model.fit([X1, X2], y, epochs=20, batch_size=64)
    print(ind)
print("Completed Training :)")


# # evaluate the model
# scores = siamese.evaluate([data_a, data_b], labels, verbose=0)
# print("%s: %.2f%%" % (siamese.metrics_names[1], scores[1] * 100))
#
# # serialize model to JSON
# model_json = model.to_json()
# with open("model_data/geyser_model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model_data/geyser_model.h5")
# print("Saved model to disk")
#
# # load json and create model
# json_file = open('model_data/geyser_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model_data/geyser_model.h5")
# print("Loaded model from disk")
#
# # evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate([data_a, data_b], labels, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))