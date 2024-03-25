import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

class Perceptron():
    def __init__(self, alpha, epochs):
        self.alpha = alpha
        self.epochs = epochs
        self.w = None
        self.train_errors = []
        self.validation_errors = []
        self.train_accuracies = []
        self.validation_accuracies = []

    def train(self, x_train, y_train, x_val=None, y_val=None):
        n_samples, n_features = x_train.shape
        self.w = np.random.rand(n_features)

        for epoch in range(self.epochs):
            # Forward pass
            y_pred_train = self.step_function(np.dot(x_train, self.w))

            # Calculate accuracy
            train_accuracy = accuracy_score(y_train, y_pred_train)
            self.train_accuracies.append(train_accuracy)

            # Update lists of errors
            train_error = 1 - train_accuracy
            self.train_errors.append(train_error)


            # Backpropagation
            error = y_train - y_pred_train

            if epoch % 10 ==0:
                print(f"Epoch {epoch} Error: {train_error}")

            # Update weights using gradient descent
            self.w += self.alpha * np.dot(x_train.T, error) / n_samples

            y_pred_val = self.step_function(np.dot(x_val, self.w))

            val_accuracy = accuracy_score(y_val, y_pred_val)
            self.validation_accuracies.append(val_accuracy)

            val_error = 1 - val_accuracy
            self.validation_errors.append(val_error)

    def step_function(self, x):
        return np.where(x >= 0, 1.0, 0.0)

    def predict(self, x):
        return self.step_function(np.dot(x, self.w))

    def plot_confusion_matrix(self, y_true, y_pred):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(y_true, y_pred,)
        cmd = ConfusionMatrixDisplay(cm,)
        cmd.plot()

    def plot_learning_curve(self):
        plt.plot(range(1, self.epochs + 1), self.train_errors, label='Training Error')
        if self.validation_accuracies:
            plt.plot(range(1, self.epochs + 1), self.validation_errors, label='Validation Error')
        plt.xlabel('Epochs')
        plt.ylabel('Errors')
        plt.title('Errors')
        plt.legend()
        plt.show()

    def plot_accuracy_curve(self):
        plt.plot(range(1, self.epochs + 1), self.train_accuracies, label='Training Accuracy')
        if self.validation_accuracies:
            plt.plot(range(1, self.epochs + 1), self.validation_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.show()