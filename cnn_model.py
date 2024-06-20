import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam  




import os

# Define the path to the dataset directory
dataset_path = "E:\Study\Bionformatics\Graduation Project\Breast Cancer\Images Model\Dataset_BUSI_with_GT"

# Get the list of image files
image_files = []
labels = []

# Iterate over the directories
for class_name in ['benign', 'malignant', 'normal']:
    class_path = os.path.join(dataset_path, class_name)
    for filename in os.listdir(class_path):
        image_path = os.path.join(class_path, filename)
        image_files.append(image_path)
        labels.append(class_name)

# Convert labels to numerical values
from tensorflow.keras.utils import to_categorical

# Convert labels to numerical values
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Convert labels to one-hot encoded format
labels = to_categorical(labels)
print(f"length of (images , labels) is ---> {labels.shape}")



import matplotlib.pyplot as plt

class_labels = ['benign', 'malignant', 'normal']

# Get the number of images in each class
class_counts = np.bincount(labels.argmax(axis=1))

# Create the bar chart
plt.bar(range(len(class_counts)), class_counts)
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.title("Class Distribution")
plt.xticks(range(len(class_labels)), class_labels)  # Set the tick labels
plt.show()


X_train, X_test, y_train, y_test = train_test_split(image_files, labels, test_size=0.2, random_state=20, stratify=labels)
print(len(X_train),len(y_train))
print(len(X_test),len(y_test)) 



# Define a function to load and preprocess the images
def load_preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.resize((150, 150))
    image = image.convert('L')
    image = np.array(image)
    image = image.reshape((150, 150, 1))
    image = image.astype('float32') / 255.0
    return image

X_train = [str(image_path) for image_path in X_train]
X_test = [str(image_path) for image_path in X_test]

# Load and preprocess the training images and labels
X_train = [load_preprocess_image(image_path) for image_path in X_train]
X_train = np.array(X_train)
y_train= np.array(y_train)

# Load and preprocess the testing images and labels
X_test= [load_preprocess_image(image_path) for image_path in X_test]
X_test = np.array(X_test)
y_test = np.array(y_test)
print(X_train.shape, y_train.shape)
print(X_test.shape,y_test.shape)




from keras import regularizers

# Initialize the model
cnn_model = Sequential()

# Add convolutional layers
cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))

# Add more convolutional layers
cnn_model.add(Conv2D(256, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))

# Flatten the output
cnn_model.add(Flatten())


# Add dense layers
cnn_model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01)))
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(64, input_dim=64,
                kernel_regularizer=regularizers.l2(0.01)))
cnn_model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))
cnn_model.add(Dense(3, activation='softmax'))

cnn_model.summary()
# Compile the model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
cnn_model.fit(X_train, y_train, batch_size=25, epochs=100, validation_split=0.1)





import numpy as np
from sklearn.metrics import confusion_matrix

# Make predictions using the model
y_pred = cnn_model.predict(X_test)

# Convert probability scores to class predictions
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Calculate confusion matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1', 'Class 2'],
            yticklabels=['Class 0', 'Class 1', 'Class 2'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
   
   
   
   
   
   
train_loss, train_accuracy = cnn_model.evaluate(X_train, y_train, verbose=0)
test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test, verbose=0)

print('Training Accuracy:', train_accuracy,"Train loss",train_loss)
print('Testing Accuracy:', test_accuracy,"Test Loss",test_loss)


from sklearn.metrics import precision_score, recall_score

# Calculate precision
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')

# Calculate recall
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')

print("Precision:", precision)
print("Recall:", recall)



from sklearn.metrics import roc_curve, auc

# Assuming you have your test labels (y_test) and predicted probabilities (y_pred)

n_classes = y_test.shape[1]  # Number of classes (assuming one-hot encoded labels)
roc_aucs = []
for i in range(n_classes):
    fpr, tpr, thresholds = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    roc_aucs.append(roc_auc)
    print(f"Class {i+1} ROC AUC: {roc_auc:.4f}")

# Calculate average ROC AUC (optional)
average_roc_auc = sum(roc_aucs) / n_classes
print(f"Average ROC AUC: {average_roc_auc:.4f}")



import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Assuming you have your test labels (y_test) and predicted probabilities (y_pred)

n_classes = y_test.shape[1]  # Number of classes (assuming one-hot encoded labels)
plt.figure(figsize=(10, 6))

for i in range(n_classes):
    fpr, tpr, thresholds = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i+1} (AUC: {roc_auc:.4f})')

plt.title('ROC Curves (One-vs-Rest)')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()




# Train the model
history = cnn_model.fit(X_train, y_train, batch_size=25, epochs=100, validation_split=0.1)

# Plot loss curve
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Plot accuracy curve
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()




