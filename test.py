import numpy as np 
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import datasets 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
from oneVsAll import LogisticRegression
# from sklearn.linear_model import LogisticRegression


def limit_class_samples_exact(data, label_column, n=1000):
    # Create an empty list to store the sampled data
    sampled_data = []

    # Iterate over each class in the label column
    for label in data[label_column].unique():
        # Select all samples from the current class
        class_data = data[data[label_column] == label]
        
        # If there are more than 'n' samples, randomly sample 'n' of them
        if len(class_data) > n:
            class_data = class_data.sample(n=n, random_state=42)
        
        # Append the selected class samples to the sampled_data list
        sampled_data.append(class_data)
    
    # Concatenate the sampled data into a single DataFrame
    return pd.concat(sampled_data).reset_index(drop=True)

data = pd.read_csv('D:\\MLAssignment1\\mlProject\\A_Z Handwritten Data.csv', header=None)
print(data)
data = limit_class_samples_exact(data , 0,1000)
print(data)
pixels = data.iloc[:,1:]
pixels = pixels / 255.0
alphabets = data.iloc[:,0]

X_train, X_temp, y_train, y_temp = train_test_split(pixels, alphabets, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.fit_transform(X_test)



dataDictionary = {
                    "A": {"inputs": X_scaled_train, "outputs": (y_train == 0).astype(float)},
                    "B": {"inputs": X_scaled_train, "outputs": (y_train == 1).astype(float)},
                    "C": {"inputs": X_scaled_train, "outputs": (y_train == 2).astype(float)},
                    "D": {"inputs": X_scaled_train, "outputs": (y_train == 3).astype(float)},
                    "E": {"inputs": X_scaled_train, "outputs": (y_train == 4).astype(float)},
                    "F": {"inputs": X_scaled_train, "outputs": (y_train == 5).astype(float)},
                    "G": {"inputs": X_scaled_train, "outputs": (y_train == 6).astype(float)},
                    "H": {"inputs": X_scaled_train, "outputs": (y_train == 7).astype(float)},
                    "I": {"inputs": X_scaled_train, "outputs": (y_train == 8).astype(float)},
                    "J": {"inputs": X_scaled_train, "outputs": (y_train ==9).astype(float)},
                    "K": {"inputs": X_scaled_train, "outputs": (y_train == 10).astype(float)},
                    "L": {"inputs": X_scaled_train, "outputs": (y_train == 11).astype(float)},
                    "M": {"inputs": X_scaled_train, "outputs": (y_train == 12).astype(float)},
                    "N": {"inputs": X_scaled_train, "outputs": (y_train == 13).astype(float)},
                    "O": {"inputs": X_scaled_train, "outputs": (y_train == 14).astype(float)},
                    "P": {"inputs": X_scaled_train, "outputs": (y_train == 15).astype(float)},
                    "Q": {"inputs": X_scaled_train, "outputs": (y_train == 16).astype(float)},
                    "R": {"inputs": X_scaled_train, "outputs": (y_train == 17).astype(float)},
                    "S": {"inputs": X_scaled_train, "outputs": (y_train == 18).astype(float)},
                    "T": {"inputs": X_scaled_train, "outputs": (y_train == 19).astype(float)},
                    "U": {"inputs": X_scaled_train, "outputs": (y_train == 20).astype(float)},
                    "W": {"inputs": X_scaled_train, "outputs": (y_train == 21).astype(float)},
                    "V": {"inputs": X_scaled_train, "outputs": (y_train == 22).astype(float)},
                    "X": {"inputs": X_scaled_train, "outputs": (y_train == 23).astype(float)},
                    "Y": {"inputs": X_scaled_train, "outputs": (y_train == 24).astype(float)},
                    "Z": {"inputs": X_scaled_train, "outputs": (y_train == 25).astype(float)},
                }
condition = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "W", "V", "X", "Y", "Z"
]


label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
clf= LogisticRegression(learningRate=0.1 , n = 10000)
clf.calculateCost(dataDictionary , condition)
pred = clf.predict(X_scaled_test)
# clf.predictForEachClass(X_scaled_test, dataDictionary,condition)

for i, cost in enumerate(clf.costValues):
    plt.plot(cost, label=f'Class {i}')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Training Cost Function Values')
plt.legend()
plt.show()


plt.plot(range(len(clf.accuracies)), clf.accuracies, label='Training Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Curve')
plt.show()

cm = confusion_matrix(y_test_encoded, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# acc = accuracy_score(y_test_encoded , pred)
# print(acc)

avg_f1 = f1_score(y_test_encoded, pred, average="weighted")
print("Average F1 Score:", avg_f1)

test_accuracy = accuracy_score(y_test_encoded, pred)
print("Test Accuracy:", test_accuracy)
