import matplotlib.pyplot as plt
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
import time

# Record start time
start_time = time.time()

# Fetch MNIST dataset
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Use 1000 samples and normalize pixel values
X = mnist.data[:5000].astype(float) / 255.0 #MUST NORMALIZE
y = mnist.target[:5000].astype(int)

print(f"X dtype: {X.dtype}, y dtype: {y.dtype}")
print(f"Unique labels in y: {np.unique(y)}")

# Plot sample images
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, X[:4], y[:4]):
    ax.set_axis_off()
    ax.imshow(image.reshape(28, 28), cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Training: {label}")
plt.show()

# Split data (with shuffling)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True, random_state=42)

# Train the basic SVC model
clf = SVC(gamma=0.001)
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)

# Plot predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test[:4], predicted[:4]):
    ax.set_axis_off()
    image = image.reshape(28, 28)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")
plt.show()

# Define parameter grid for GridSearchCV
svm_new = SVC()
param_grid = { # try to find a trend for the parameters, small dataset is still fine
    'C': np.linspace(0.05,0.1,10), 
    'kernel': ['rbf'], 
    'gamma': np.linspace(0.05, 1, 10)
}

# Grid search with cross-validation
grid_search = GridSearchCV(svm_new, param_grid, cv=3)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
accuracy = accuracy_score(y_test, predicted)
print(accuracy)

# Calculate elapsed time
elapsed_time = time.time() - start_time
print(f"Script executed in {elapsed_time:.4f} seconds")
