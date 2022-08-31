import pickle as pk

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC


def print_metrics(y_val, y_pred):
    cm = confusion_matrix(y_val, y_pred)
    accuracy = float(cm.diagonal().sum()) / len(y_val)
    print("\nAccuracy Of SVM For The Given Dataset: ", accuracy)
    print('\nConfusion Matrix:\n', cm)
    print('\nClassification report:\n', classification_report(y_val, y_pred))


class FeatureClassifier:
    def __init__(self, classifier=SVC(kernel='rbf', probability=True)):
        self.model = classifier

    def fit(self, X_train, y_train, X_val):
        self.model.fit(X_train, y_train)
        return self.predict(X_val)

    def predict(self, X_val):
        print("\nWait for prediction by SVM...")
        y_pred = self.model.predict(X_val)
        return y_pred

    def predict_proba(self, X_val):
        print("\nWait for probabilities prediction by SVM...")
        y_pred = self.model.predict_proba(X_val)
        return y_pred

    def save(self, path_fitted_model, output_path='/predictor.pkl'):
        with open(path_fitted_model + output_path, 'wb') as file:
            pk.dump(self.model, file)

    def load(self, path_fitted_model, input_path='/predictor.pkl'):
        with open(path_fitted_model + input_path, 'rb') as file:
            self.model = pk.load(file)
