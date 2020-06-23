import os 
import operator
import numpy as np
import pandas as pd 
from time import time
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import metrics
from sklearn  import feature_selection

import tensorflow as tf
from tensorflow import keras

from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler

from utilities import Utilities 

from mlxtend.evaluate import bias_variance_decomp

class ModelRunner: 

    def __init__(self, base_folder_address):
        super().__init__()
        self.base_folder_address = base_folder_address

    def drive_model_runner(self):

        columns_from_java_output = ['name', 'hyphens', 'colons', 'semi_colon', 'interjections', 'male_oriented', 'female_oriented', 'comma', 'period', 'fres']

        name_of_python_output = Utilities.get_prop_value(Utilities.PYTHON_FEATURE_CSV)
        name_of_java_output = Utilities.get_prop_value(Utilities.JAVA_FEATURE_CSV)

        path_to_python_out = os.path.join(os.getcwd(), name_of_python_output)
        path_to_java_out = os.path.join(os.getcwd(), name_of_java_output)

        python_out = pd.read_csv(path_to_python_out)
        java_out = pd.read_csv(path_to_java_out)

        python_out.drop(['avg_punctuation_per_sentence'], axis=1, inplace=True)
        java_out = java_out[columns_from_java_output]

        df = python_out.merge(java_out, left_on='book_id', right_on='name')
        df.drop(['name'], axis=1, inplace=True)

        cols = list(df.columns.values)
        cols.pop(cols.index('genre'))
        df = df[cols+['genre']]

        df.drop(df.loc[df['genre'] == 'Allegories'].index, inplace=True)

        df['genre'] = df['genre'].astype('category')
        df['genre'] = df['genre'].cat.codes

        X = df.iloc[:, 1:-1].to_numpy()
        Y = df.iloc[:, -1].to_numpy()

        multi_model_start_time = time()
        svm_acc_poly, svm_acc_rbf, nb_acc, lr_acc, rf_acc, en_acc = self.run_multiple_model(X, Y)
        multi_model_end_time = time()

        nn_start_time = time()
        nn_acc = self.run_nn(X, Y)
        nn_end_time = time()
        
        val_acc_svm_poly, test_svm_poly = svm_acc_poly
        val_acc_svm_rbf, test_svm_rbf = svm_acc_rbf
        val_acc_nb, test_nb = nb_acc
        val_acc_logistic, test_logistic = lr_acc
        val_acc_rf, test_rf = rf_acc
        val_acc_ensemble, test_ensemble = en_acc

        print("Performance Report")
        print("--------------------------------------")
        print("SVM With PolyKernel Report")
        print("validation accuracy: {}, test accuracy: {}".format(val_acc_svm_poly, test_svm_poly))
        
        print("SVM With RBF-Kernel Report")
        print("validation accuracy: {}, test accuracy: {}".format(val_acc_svm_rbf, test_svm_rbf))
        
        print("Multi-NB Report")
        print("validation accuracy: {}, test accuracy: {}".format(val_acc_nb, test_nb))
        
        print("Multi-Logistic Report")
        print("validation accuracy: {}, test accuracy: {}".format(val_acc_logistic, test_logistic))

        print("Random-Forrest Report")
        print("validation accuracy: {}, test accuracy: {}".format(val_acc_rf, test_rf))

        print("Ensemble Report")
        print("validation accuracy: {}, test accuracy: {}".format(val_acc_ensemble, test_ensemble))

        print("Neural Net Report")
        print("Accuracy: {}".format(nn_acc))
        
        print("Timing Report")
        print("--------------------------------------")
        print("Neural-Network : {} minutes".format((nn_end_time-nn_start_time)/60))
        print("Rest of the algo : {} minutes".format((multi_model_end_time-multi_model_start_time)/60))
        
        return

    def convert_data(self, a, b):
    
        row_list =[]     

        for index, rows in a.iterrows(): 
            
            my_list = [rows[i] for i in a.columns]     
            row_list.append(my_list) 
        
        target = [rows for _, rows in b.items()]
            
        return row_list, target
    
    def run_nn(self, X, Y):

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state = 0, shuffle=True, stratify = Y)

        X_train, Y_train = self.tackle_data_imbalance(X_train, Y_train)

        train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_data = train_data.shuffle(buffer_size=500).batch(50).repeat(200)

        test_data = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        test_data = test_data.batch(64)

        model = keras.models.Sequential([
            keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
            keras.layers.Dense(50, activation="relu"),
            keras.layers.Dense(50, activation="relu"),
            keras.layers.Dense(1)
        ])
        
        model.compile(loss='categorical_crossentropy', 
            metrics=['accuracy'], 
            optimizer=keras.optimizers.RMSprop(lr=1e-3)
        )

        model.fit(train_data, epochs=10)

        result = model.evaluate(test_data)        

        return result[1]
        
    def run_multiple_model(self, X, Y):

        n_split = 3
        val_acc_svm_poly = 0
        val_acc_svm_rbf = 0
        val_acc_logistic = 0
        val_acc_nb = 0
        val_acc_rf = 0
        val_acc_ensemble = 0

        X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.25, random_state = 0, shuffle=True, stratify = Y)

        # selecting 20 best features
        X_train_val, X_test = self.select_k_features(k=20, X_train=X_train_val, X_test=X_test, Y_train=Y_train_val)

        self.bias_variance_decomp(X_train_val, X_test, Y_train_val, Y_test)

        skf = StratifiedKFold(n_splits=n_split, shuffle=True)

        for train_index, val_index in skf.split(X_train_val, Y_train_val):
            
            X_train, X_val = X_train_val[train_index], X_train_val[val_index]
            Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]

            X_train, Y_train = self.tackle_data_imbalance(X_train, Y_train)
            
            val_acc_svm_poly += self.perform_SVM_with_Polynomial_kernel(X_train, Y_train, X_val, Y_val)
            val_acc_svm_rbf += self.perform_SVM_with_RBF_kernel(X_train, Y_train, X_val, Y_val)
            val_acc_logistic += self.perform_Logistic(X_train, Y_train, X_val, Y_val)
            val_acc_nb += self.perform_NB(X_train, Y_train, X_val, Y_val)    
            val_acc_rf += self.perform_random_forrest(X_train, Y_train, X_val, Y_val)
            val_acc_ensemble += self.do_ensemble_and_learn(X_train, Y_train, X_val, Y_val)

        val_acc_svm_poly = float(val_acc_svm_poly/n_split)
        val_acc_svm_rbf = float(val_acc_svm_rbf/n_split)
        val_acc_logistic = float(val_acc_logistic/n_split)
        val_acc_nb = float(val_acc_nb/n_split) 
        val_acc_rf = float(val_acc_rf/n_split) 
        val_acc_ensemble = float(val_acc_ensemble/n_split) 
        
        test_svm_poly = self.perform_SVM_with_Polynomial_kernel(X_train, Y_train, X_test, Y_test)
        test_svm_rbf = self.perform_SVM_with_RBF_kernel(X_train, Y_train, X_test, Y_test)
        test_logit = self.perform_Logistic(X_train, Y_train, X_test, Y_test)
        test_nb = self.perform_NB(X_train, Y_train, X_test, Y_test)
        test_rf = self.perform_random_forrest(X_train, Y_train, X_test, Y_test)
        test_ensemble = self.do_ensemble_and_learn(X_train, Y_train, X_test, Y_test)

        return (val_acc_svm_poly, test_svm_poly), (val_acc_svm_rbf, test_svm_rbf), (val_acc_nb, test_nb), (val_acc_logistic, test_logit), (val_acc_rf, test_rf), (val_acc_ensemble, test_ensemble)
        

    def perform_SVM_with_Polynomial_kernel(self, X_train, Y_train, X_val, Y_val):
        
        svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="poly", degree=10, coef0=50, C=500))
        ])
        
        svm_clf.fit(X_train, Y_train)
        Y_pred_svm = svm_clf.predict(X_val)
        
        return metrics.accuracy_score(Y_val, Y_pred_svm)
    
    def perform_SVM_with_RBF_kernel(self, X_train, Y_train, X_val, Y_val):
        
        svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=0.0001, C=0.1))
        ])
        
        svm_clf.fit(X_train, Y_train)
        Y_pred_svm = svm_clf.predict(X_val)
        
        return metrics.accuracy_score(Y_val, Y_pred_svm)

    def perform_Logistic(self, X_train, Y_train, X_val, Y_val):
        
        
        clf = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(multi_class="multinomial", solver="lbfgs", C=50, n_jobs=-1, class_weight='balanced', max_iter=10000))
            ])        
                        
        clf.fit(X_train, Y_train)
        Y_pred_logistic = clf.predict(X_val)
        return metrics.accuracy_score(Y_val, Y_pred_logistic)

        # Multinomial Naive Bayes

    def perform_NB(self, X_train, Y_train, X_val, Y_val):

        nb = MultinomialNB(alpha=0.001)
        nb.fit(X_train, Y_train)
        Y_pred_nb = nb.predict(X_val)
        return metrics.accuracy_score(Y_val, Y_pred_nb)

    def perform_random_forrest(self, X_train, Y_train, X_val, Y_val):

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier())
        ])
        
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_val)
        return metrics.accuracy_score(Y_val, Y_pred)

    def select_k_features(self, k, X_train, X_test, Y_train):
        ch2 = feature_selection.SelectKBest(feature_selection.chi2, k=k)
        X_train = ch2.fit_transform(X_train, Y_train)
        X_test = ch2.transform(X_test)
        return X_train, X_test    
    
    def tackle_data_imbalance(self, X, Y):
        
        increase = 3
        counter = Counter(Y)
        
        total_classes = len(counter)
        total_data_points = sum(counter.values())
        expected_points = total_data_points*increase
        avg_points_per_class = int(expected_points/total_classes)
        
        # generating highest amount of data for each class 
        # higest_key, highest_val = max(counter.items(), key=operator.itemgetter(1))
        # famous_dict = dict((key, highest_val) for key in counter) 
        
        famous_dict = dict((key, avg_points_per_class) for key in counter) # generating double of previous for each class
        
        over = ADASYN(n_neighbors=1, sampling_strategy=famous_dict)    
        under = RandomUnderSampler(sampling_strategy="auto")
        
        X, Y = over.fit_resample(X, Y)
        X, Y = under.fit_resample(X, Y)
        return X, Y

    def do_ensemble_and_learn(self, X_train, Y_train, X_test, Y_test): 

        clf1 = ("svm_clf", SVC(kernel="poly", degree=10, coef0=50, C=500))
        clf2 = ("lr", LogisticRegression(multi_class="multinomial", solver="lbfgs", C=50, n_jobs=-1, max_iter=10000))
        clf3 = ("rf", RandomForestClassifier())
        
        ensemble_clf = VotingClassifier(
                    estimators=[
                        clf1, 
                        clf2,
                        clf3
                    ], 
                    voting='hard'
                )

        eclf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ensemble_clf)
        ])

        eclf = eclf.fit(X_train, Y_train)
        Y_preds = eclf.predict(X_test)
        return metrics.accuracy_score(Y_test, Y_preds)

    def bias_variance_decomp(self, X_train, X_test, Y_train, Y_test):
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X=X_train)
        X_test = scaler.transform(X_test)
        rf = RandomForestClassifier()        

        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
                rf, X_train, Y_train, X_test, Y_test, 
                loss='0-1_loss')

        print('Decomposing Bias and Variance of RandomForest')
        print('-------------------------------------------')
        print('Average expected loss: %.3f' % avg_expected_loss)
        print('Average bias: %.3f' % avg_bias)
        print('Average variance: %.3f' % avg_var)
        print('-------------------------------------------')
        return