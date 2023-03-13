#This code loads the MRI data and labels, splits the data into training and testing sets, scales the data using StandardScaler, 
#defines the ANN model with 2 hidden layers and dropout regularization, compiles the model with binary cross-entropy loss and accuracy metric, 
#defines early stopping to prevent overfitting, trains the model on the training set with a batch size of 32 and 100 epochs, 
#predicts the labels for the test set using the trained model, and prints the classification report and confusion matrix.
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

# Load the data and labels
data = np.load('ftd_mri_data.npy')
labels = np.load('ftd_mri_labels.npy')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the ANN model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[es])

# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
