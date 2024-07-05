import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
#from sklearn.metrics import roc_curve, roc_auc_score

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
            self.learning_rate = learning_rate
            self.num_iterations = num_iterations
            self.weights = None
            self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls
    
def app():
    col1, col2, col3 = st.columns([0.5,4,0.2])
    with col2:
        st.title("Training Model and Score")
    def drop_name_column_if_exists(df):
    # Check if 'name' column exists and drop it
        if 'name' in df.columns:
            df = df.drop(columns=['name'])
        return df
    uploaded_file = st.file_uploader("Please Choose the dataset", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = drop_name_column_if_exists(df)
        X = df.drop(columns=['status'], axis=1)
        Y = df['status']

        def train_test_split(X, Y, test_size=0.2, random_state=2):
            if isinstance(X, pd.DataFrame):
                X = X.values  # Convert DataFrame to NumPy array

            if random_state is not None:
                np.random.seed(random_state)

            n_samples = X.shape[0]
            n_test = int(n_samples * test_size)

            indices = np.random.permutation(n_samples)
            train_indices, test_indices = indices[n_test:], indices[:n_test]

            X_train, X_test = X[train_indices], X[test_indices]
            Y_train, Y_test = Y[train_indices], Y[test_indices]

            return X_train, X_test, Y_train, Y_test

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

        class MinMaxScaler:
            def __init__(self, feature_range=(0, 1)):
                self.feature_range = feature_range

            def fit(self, X):
                self.data_min = np.min(X, axis=0)
                self.data_max = np.max(X, axis=0)
                self.data_range = self.data_max - self.data_min
                self.scale = (self.feature_range[1] - self.feature_range[0]) / (self.data_range + np.finfo(float).eps)

            def transform(self, X):
                X_scaled = (X - self.data_min) * self.scale + self.feature_range[0]
                return X_scaled

            def fit_transform(self, X):
                self.fit(X)
                return self.transform(X)

        minmax = MinMaxScaler()
        X_train_scaled = minmax.fit_transform(X_train)
        X_test_scaled = minmax.transform(X_test)

        def accuracy_score(y_true, y_pred):
            correct = 0
            total = len(y_true)
            for true, pred in zip(y_true, y_pred):
                if true == pred:
                    correct += 1
            accuracy = correct / total
            return accuracy

        def cross_val_score(model, X, y, cv):
            n = len(X)
            fold_size = n // cv
            scores = []
            for i in range(cv):
                start = i * fold_size
                end = (i + 1) * fold_size
                X_test = X[start:end]
                y_test = y[start:end]
                X_train = np.concatenate([X[:start], X[end:]])
                y_train = np.concatenate([y[:start], y[end:]])
                model.fit(X_train, y_train)
                score = accuracy_score(y_test, model.predict(X_test))
                scores.append(score)
            return scores

        def precision_score(y_true, y_pred):
            true_positives = 0
            total_positives_predicted = 0
            for true, pred in zip(y_true, y_pred):
                if pred == 1:
                    total_positives_predicted += 1
                    if true == pred:
                        true_positives += 1
            precision = true_positives / total_positives_predicted if total_positives_predicted > 0 else 0
            return precision

        def recall_score(y_true, y_pred):
            true_positives = 0
            actual_positives = 0
            for true, pred in zip(y_true, y_pred):
                if true == 1:
                    actual_positives += 1
                    if true == pred:
                        true_positives += 1
            recall = true_positives / actual_positives if actual_positives > 0 else 0
            return recall

        def f1_score(y_true, y_pred):
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            return f1

        def confusion_matrix(y_true, y_pred):
            true_positive = sum((true == 1 and pred == 1) for true, pred in zip(y_true, y_pred))
            false_positive = sum((true == 0 and pred == 1) for true, pred in zip(y_true, y_pred))
            true_negative = sum((true == 0 and pred == 0) for true, pred in zip(y_true, y_pred))
            false_negative = sum((true == 1 and pred == 0) for true, pred in zip(y_true, y_pred))
            return [[true_positive, false_positive], [false_negative, true_negative]]

        def roc_curve(y_true, y_scores, pos_label=1):
            fpr = []
            tpr = []
            thresholds = []
    
            # Sort by score
            desc_score_indices = np.argsort(y_scores)[::-1]
            y_true = np.array(y_true)[desc_score_indices]
            y_scores = np.array(y_scores)[desc_score_indices]
    
            # Calculate the number of positive and negative examples
            P = np.sum(y_true == pos_label)
            N = len(y_true) - P
    
            TP = 0
            FP = 0
            prev_score = float('inf')
    
            for i in range(len(y_true)):
                if y_scores[i] != prev_score:
                    fpr.append(FP / N)
                    tpr.append(TP / P)
                    thresholds.append(y_scores[i])
                    prev_score = y_scores[i]
        
                if y_true[i] == pos_label:
                    TP += 1
                else:
                    FP += 1
    
            # Add the last point (1,1)
            fpr.append(1)
            tpr.append(1)
            thresholds.append(0)
    
            return np.array(fpr), np.array(tpr), np.array(thresholds)
        
        def roc_auc_score(y_true, y_scores):
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            return np.trapz(tpr, fpr)
        
        def Evaluate_Performance(Model, Xtrain, Xtest, Ytrain, Ytest):
            Model.fit(Xtrain, Ytrain)
            overall_score = cross_val_score(Model, Xtrain, Ytrain, cv=10)
            model_score = np.average(overall_score)
            Ypredicted = Model.predict(Xtest)
            training_accuracy = accuracy_score(Ytrain, Model.predict(Xtrain))
            st.markdown("*Performance of the Model(Logistic Regression)*")
            # Prepare the scores as a DataFrame for better visualization
            scores_data = {
                "Performance": ["Training Accuracy", "**Cross Validation**", "Testing Accuracy", "Precision", "Recall", "F1-Score"],
                "Score": [
                    round(training_accuracy * 100, 2),
                    round(model_score * 100, 2),
                    round(accuracy_score(Ytest, Ypredicted) * 100, 2),
                    round(precision_score(Ytest, Ypredicted) * 100, 2),
                    round(recall_score(Ytest, Ypredicted) * 100, 2),
                    round(f1_score(Ytest, Ypredicted) * 100, 2)
                ]
            }
            scores_df = pd.DataFrame(scores_data)
            st.table(scores_df)  
            conf_matrix = confusion_matrix(Ytest, Ypredicted)
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=plt.cm.Blues, annot_kws={"size": 16})
            plt.title('Confusion Matrix', y=1.05, fontsize=20, fontfamily='Times New Roman')
            plt.xlabel('Predicted Labels', labelpad=25, fontsize=15, fontfamily='Times New Roman')
            plt.ylabel('True Labels', labelpad=25, fontsize=15, fontfamily='Times New Roman')
            st.pyplot(plt)
            
            # ROC Curve
            y_pred_proba = Model.predict_proba(Xtest)
            fpr, tpr, _ = roc_curve(Ytest, y_pred_proba)
            roc_auc = roc_auc_score(Ytest, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            st.pyplot(plt)

        

        LR = LogisticRegression()
        LR.fit(X_train_scaled, Y_train)
        Y_pred_LR = LR.predict(X_test_scaled)
        with open('logistic_regression_model.pkl', 'wb') as file:
            pickle.dump(LR, file)
        Evaluate_Performance(LR, X_train_scaled, X_test_scaled, Y_train, Y_test)
        
  
    
    
