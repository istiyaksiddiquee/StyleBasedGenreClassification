import os 
import operator
import numpy as np
import pandas as pd 
from time import time
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from sklearn  import feature_selection

import tensorflow as tf
from tensorflow import keras

from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import RandomUnderSampler

from utilities import Utilities 

class ModelRunner: 

    def __init__(self, base_folder_address):
        super().__init__()
        self.base_folder_address = base_folder_address

    def drive_model_runner(self):

        path_to_file = os.path.join(os.getcwd(), Utilities.get_prop_value(Utilities.FEATURE_CSV_KEY))        
        df = pd.read_csv(path_to_file, encoding=Utilities.get_file_encoding(path_to_file=path_to_file))

        df.drop(df.loc[df['genre'] == 'Allegories'].index, inplace=True)

        df['genre'] = df['genre'].astype('category')
        df['genre'] = df['genre'].cat.codes

        X = df.iloc[:, 1:-1].to_numpy()
        Y = df.iloc[:, -1].to_numpy()        

        multi_model_start_time = time()
        svm_acc, nb_acc, lr_acc, rf_acc = self.run_multiple_model(X, Y)
        multi_model_end_time = time()

        nn_start_time = time()
        nn_acc = self.run_nn(X, Y)
        nn_end_time = time()
        
        val_acc_svm, test_svm = svm_acc
        val_acc_nb, test_nb = nb_acc
        val_acc_logistic, test_logistic = lr_acc
        val_acc_rf, test_rf = rf_acc

        print("Performance Report")
        print("--------------------------------------")
        print("SVM Report")
        print("validation accuracy: {}, test accuracy: {}".format(val_acc_svm, test_svm))
        
        print("Multi-NB Report")
        print("validation accuracy: {}, test accuracy: {}".format(val_acc_nb, test_nb))
        
        print("Multi-Logistic Report")
        print("validation accuracy: {}, test accuracy: {}".format(val_acc_logistic, test_logistic))

        print("Random-Forrest Report")
        print("validation accuracy: {}, test accuracy: {}".format(val_acc_rf, test_rf))

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

        val_acc_svm = 0
        val_acc_logistic = 0
        val_acc_nb = 0
        val_acc_rf = 0

        X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.20, random_state = 0, shuffle=True, stratify = Y)

        # selecting 10 best features
        X_train_val, X_test = self.select_k_features(k=10, X_train=X_train_val, X_test=X_test, Y_train=Y_train_val)

        skf = StratifiedKFold(n_splits=5, shuffle=True)

        for train_index, val_index in skf.split(X_train_val, Y_train_val):
            
            X_train, X_val = X_train_val[train_index], X_train_val[val_index]
            Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]

            X_train, Y_train = self.tackle_data_imbalance(X_train, Y_train)
            
            val_acc_svm += self.perform_SVM(X_train, Y_train, X_val, Y_val)
            val_acc_logistic += self.perform_Logistic(X_train, Y_train, X_val, Y_val)
            val_acc_nb += self.perform_NB(X_train, Y_train, X_val, Y_val)    
            val_acc_rf += self.perform_random_forrest(X_train, Y_train, X_val, Y_val)        

        val_acc_svm = float(val_acc_svm/5)
        val_acc_logistic = float(val_acc_logistic/5)
        val_acc_nb = float(val_acc_nb/5) 
        val_acc_rf = float(val_acc_rf/5) 
        
        test_svm = self.perform_SVM(X_train, Y_train, X_test, Y_test)
        test_logit = self.perform_Logistic(X_train, Y_train, X_test, Y_test)
        test_nb = self.perform_NB(X_train, Y_train, X_test, Y_test)
        test_rf = self.perform_random_forrest(X_train, Y_train, X_test, Y_test)

        return (val_acc_svm, test_svm), (val_acc_nb, test_nb), (val_acc_logistic, test_logit), (val_acc_rf, test_rf)
        

    def perform_SVM(self, X_train, Y_train, X_val, Y_val):
        
        # SVM 
        
        svm_clf = Pipeline([        
            ("linear_svc", LinearSVC(C=1, loss="hinge", max_iter=-1)),
        ])

        svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="poly", degree=10, coef0=50, C=500))
        ])


        # ("svm_clf", SVC(kernel="rbf", gamma=0.0001, C=0.1)) -> 28 
        # ("svm_clf", SVC(kernel="poly", degree=10, coef0=50, C=500)) -> 30

        # print("fitting data to SVM")
        # svm_clf = SVC(gamma='auto')
        svm_clf.fit(X_train, Y_train)
        Y_pred_svm = svm_clf.predict(X_val)
        
        return metrics.accuracy_score(Y_val, Y_pred_svm)

    def perform_Logistic(self, X_train, Y_train, X_val, Y_val):
        
        softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, max_iter=1000)
        # print("fitting data to Logistic")
        softmax_reg.fit(X_train, Y_train)
        Y_pred_logistic = softmax_reg.predict(X_val)    
        return metrics.accuracy_score(Y_val, Y_pred_logistic)

        # Multinomial Naive Bayes

    def perform_NB(self, X_train, Y_train, X_val, Y_val):

        nb = MultinomialNB()
        # print("fitting data to Naive Bayes")
        nb.fit(X_train, Y_train)
        Y_pred_nb = nb.predict(X_val)
        return metrics.accuracy_score(Y_val, Y_pred_nb)

    def perform_random_forrest(self, X_train, Y_train, X_val, Y_val):

        clf = RandomForestClassifier(max_depth=2, random_state=0)
        clf.fit(X_train, Y_train)

        Y_pred = clf.predict(X_val)
        return metrics.accuracy_score(Y_val, Y_pred)

    def select_k_features(self, k, X_train, X_test, Y_train):
        ch2 = feature_selection.SelectKBest(feature_selection.chi2, k=k)
        X_train = ch2.fit_transform(X_train, Y_train)
        X_test = ch2.transform(X_test)
        return X_train, X_test
    
    
    def tackle_data_imbalance(self, X, Y):
        
        counter = Counter(Y)
        
        total_classes = len(counter)
        total_data_points = sum(counter.values())
        expected_points = total_data_points*3
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