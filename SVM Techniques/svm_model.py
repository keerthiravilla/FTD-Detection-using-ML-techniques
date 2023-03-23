# Import required libraries
import numpy as np
from sklearn import svm, metrics, preprocessing
from sklearn.model_selection import train_test_split
import nibabel as nib
import os

# Load the MRI image data
data_dir = 'path/to/mri/data/directory'
subjects = os.listdir(data_dir)

# Extract the MRI data and labels
X = []
y = []
for subject in subjects:
    img_path = os.path.join(data_dir, subject)
    img = nib.load(img_path)
    data = img.get_fdata()
    X.append(data.reshape(-1))
    if 'ftd' in subject:
        y.append(1)
    else:
        y.append(0)

# Preprocess the MRI data
X = preprocessing.StandardScaler().fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the SVM model
svm_model = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)

# Predict the labels for the test data
y_pred = svm_model.predict(X_test)

# Evaluate the model performance
accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)

print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
