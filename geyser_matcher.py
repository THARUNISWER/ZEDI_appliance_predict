from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Dot, Lambda, Concatenate
import keras.backend as K
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers import Dense
import pandas as pd



max_len = 3
# Define the input layers
input_a = Input(shape=(max_len,))
input_b = Input(shape=(max_len,))

# Define the embedding layer
vocab_size = 16
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

# Train the model on the dataset of similar and dissimilar pairs of lists
# Define the training data
data_a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
data_b = [[1, 2, 3], [10, 12, 12], [13, 14, 14]]

# Define the labels (1 for similar, 0 for dissimilar)
labels = [1, 0, 0]

# Convert the data to NumPy arrays
data_a = np.array(data_a)
data_b = np.array(data_b)
labels = np.array(labels)
num_epochs = 20
batch_size = 64

model.fit([data_a, data_b], labels, epochs=num_epochs, batch_size=batch_size)

# evaluate the model
scores = model.evaluate([data_a, data_b], labels, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

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
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate([data_a, data_b], labels, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))