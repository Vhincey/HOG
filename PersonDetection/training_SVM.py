from skimage.feature import hog
import joblib
import glob
import os
import cv2
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

train_data = []
train_labels = []
pos_im_path = 'PersonDetection/DATAIMAGE/positive'
neg_im_path = 'PersonDetection/DATAIMAGE/negative'
model_path = 'PersonDetection/models/models.dat'

# Create the models directory if it doesn't exist
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Load the positive features
for filename in glob.glob(os.path.join(pos_im_path, "*.png")):
    print(f"Processing {filename}")
    fd = cv2.imread(filename, 0)
    if fd is not None:
        fd = cv2.resize(fd, (64, 128))
        fd = hog(fd, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
        train_data.append(fd)
        train_labels.append(1)
    else:
        print(f"Unable to read image: {filename}")

# Load the negative features
print("Loading negative images:")
for filename in glob.glob(os.path.join(neg_im_path, "*.jpg")):
    print(f"Processing {filename}")
    fd = cv2.imread(filename, 0)
    if fd is not None:
        fd = cv2.resize(fd, (64, 128))
        fd = hog(fd, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
        train_data.append(fd)
        train_labels.append(0)
    else:
        print(f"Unable to read image: {filename}")

train_data = np.array(train_data)
train_labels = np.array(train_labels)

print('Data Prepared........')
print('Train Data Shape:', train_data.shape)
print('Train Labels (1,0):', len(train_labels))
print("""
Classification with SVM
""")

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

if len(X_train) > 0:
    # Initialize and train the SVM model
    model = LinearSVC(max_iter=10000)
    print('Training...... Support Vector Machine')
    model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(model, os.path.join(models_dir, 'models.dat'))
    print('Model saved: {}'.format(os.path.join(models_dir, 'models.dat')))

    # Model evaluation on the validation set
    y_pred = model.predict(X_valid)
    accuracy = metrics.accuracy_score(y_valid, y_pred)
    precision = metrics.precision_score(y_valid, y_pred)
    recall = metrics.recall_score(y_valid, y_pred)
    f1_score = metrics.f1_score(y_valid, y_pred)

    print('Model Evaluation on Validation Set:')
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1 Score:', f1_score)
else:
    print('No training data available. Check the paths to positive and negative images.')
