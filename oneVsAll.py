import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
#sigmoid function :- 
'''
Z = w * x + b 
p(i) = 1/ 1+ e^-Z
'''
def sigmoid(w , X , b): 
    Z = np.dot(X,w) +  b
    return 1/(1 + np.exp(-Z))

#cost Function :- 
'''
J = - 1/m [sum{i = 1 to m} (y * log(p(i)) + (1-y) * log(1-p(i)) ) ]
'''


def costFunction(m, hypothesis, Y):
    # Add epsilon to prevent log(0)
    epsilon = 1e-15
    hypothesis = np.clip(hypothesis, epsilon, 1 - epsilon)  # Ensure hypothesis values are in (epsilon, 1 - epsilon)
    return -np.sum(Y * np.log(hypothesis) + (1 - Y) * np.log(1 - hypothesis)) / m



    
#gradient descent :- 
'''
dw = 1/m [sum{i = 1 to m} (p(i) - y) * x]
db = 1/m [sum{i = 1 to m} (p(i) - y)]
w = w - learningRate * dw 
b = b - learningRate * db
'''

def gradienDescent(m , hypothesis , X , Y , w , b , learningRate ):
    error = hypothesis - Y
    
    dw = (1/m) * np.dot(X.T,error)
    db = (1/m) * np.sum(error)
    
    w = w - learningRate * dw
    b = b - learningRate * db
    return w , b




class LogisticRegression():
    costValues = []
    parametersTrained = []


    def __init__(self , learningRate = 0.1 , n = 10000 , features = 784, classes = 26):
        self.learningRate = learningRate
        self.n = n 
        self.classes = classes
        self.features = features
        self.accuracies = []

        
    def calculateCost(self,dataDictionary , condition):
         for i in range(26):
            print("train for dataset" , i)
            X = dataDictionary[condition[i]]["inputs"]
            Y = dataDictionary[condition[i]]["outputs"]
            self.weights = np.zeros(self.features) * self.learningRate
            m = X.shape[0]
            # print(m)
            # print(self.weights)
            self.bias = 0 
            costFunctionValuesForEachClass = []
            counter = 0 
            cost_prev = None
            for j in range(self.n):
                    hypothesis = sigmoid(self.weights , X , self.bias) # => values from 0 to 1
                    
                    cost = costFunction(m , hypothesis , Y) # calculate the cost function for every class
                    
                    costFunctionValuesForEachClass.append(cost) 
                    
                    self.weights , self.bias = gradienDescent(m , hypothesis , X , Y , self.weights ,self.bias , self.learningRate)
                    train_preds = (hypothesis >= 0.5).astype(int)
                    train_accuracy = accuracy_score(Y, train_preds)
                    self.accuracies.append(train_accuracy)
                    counter+= 1
                    
                    if (j % 100) == 0:
                        print(f"Cost at iteration {j}: {cost}")
                    if cost_prev is not None:
                        cost_diff = abs(cost - cost_prev)
                        if cost_diff < 0.000001:
                            print("Cost has converged.")
                            break 
                    cost_prev = cost
            self.costValues.append(costFunctionValuesForEachClass)
            
            self.parametersTrained.append([self.weights , self.bias])# best optimal weight and bias
            # print("counter : "  , counter)


    def predictForEachClass(self , X, dataDictionary , condition):
        X = np.array(X)
        for i in range(26):
            X = dataDictionary[condition[i]]["inputs"]
            m = X.shape[0]
            # print(m)
            Y = dataDictionary[condition[i]]["outputs"]
            weights, bias = self.parametersTrained[i]
            correct_pred = 0 
            # print(Y)
            hypothsis = sigmoid(weights , X ,bias)
            z = 0
            for h in hypothsis:
                if h >= 0.5 : 
                    if Y.iloc[z] == 1 :
                        correct_pred += 1
                if h < 0.5 : 
                    if Y.iloc[z] == 0 :
                        correct_pred+=1
                z+=1  
            acc = (correct_pred / m)*100
            print("accuracy for class " ,i , " is : " ,acc )

    def predict(self , X):
        X = np.array(X)
        prob = []
        for i in range(len(self.parametersTrained)):
            weights, bias = self.parametersTrained[i]
            # print(weights , bias)
            hypothesis = sigmoid(weights, X , bias)
            # print(hypothesis)
            prob.append(hypothesis)
        prob = np.vstack(prob)
        # print("Class probabilities:\n", prob)
        predictions = np.argmax(prob, axis=0)
        # print("Predicted class indices:", predictions)  
        return predictions
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='weighted')

        print(f"Accuracy: {acc}")
        print(f"F1 Score: {f1}")

        # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

        return acc, f1