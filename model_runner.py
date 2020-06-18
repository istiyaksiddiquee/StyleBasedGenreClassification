import os 
from time import time
import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn import metrics

import tensorflow as tf
from tensorflow import keras

from utilities import Utilities 

class ModelRunner: 

    def __init__(self, base_folder_address):
        super().__init__()
        self.base_folder_address = base_folder_address

    def drive_model_runner(self):

        path_to_file = os.path.join(os.getcwd(), Utilities.get_prop_value(Utilities.FEATURE_CSV_KEY))        
        df = pd.read_csv(path_to_file, encoding=Utilities.get_file_encoding(path_to_file=path_to_file))

        df['genre'] = df['genre'].astype('category')
        df['genre'] = df['genre'].cat.codes

        X = df.iloc[:, 1:-1].to_numpy()
        Y = df.iloc[:, -1].to_numpy()        

        multi_model_start_time = time()
        svm_acc, nb_acc, lr_acc = self.run_multiple_model(X, Y)
        multi_model_end_time = time()

        nn_start_time = time()
        nn_acc = self.run_nn(X, Y)
        nn_end_time = time()
        
        print("SVM Report")
        print("validation accuracy: {}, test accuracy: {}".format(svm_acc))


        print("Average Validation Accuracy")
        print("SVM: {}; Logistic: {}; NB: {}".format(round(val_acc_svm, 4), round(val_acc_logistic, 4), round(val_acc_nb, 4)))

        print("Test Accuracy")
        print("SVM: {}; Logistic: {}; NB: {}".format(round(test_svm, 4), round(test_logit, 4), round(test_nb, 4)))

        print("Test Accuracy")
        print("SVM: {}; Logistic: {}; NB: {}".format(round(test_svm, 4), round(test_logit, 4), round(test_nb, 4)))



        
        # print("Total time : {} minutes".format(total/60))
        
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

        X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.20, random_state = 0, shuffle=True, stratify = Y)

        skf = StratifiedKFold(n_splits=5, shuffle=True)

        for train_index, val_index in skf.split(X_train_val, Y_train_val):
            
            X_train, X_val = X_train_val[train_index], X_train_val[val_index]
            Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]
            
            val_acc_svm += self.perform_SVM(X_train, Y_train, X_val, Y_val)
            val_acc_logistic += self.perform_Logistic(X_train, Y_train, X_val, Y_val)
            val_acc_nb += self.perform_NB(X_train, Y_train, X_val, Y_val)            

        val_acc_svm = float(val_acc_svm/5)
        val_acc_logistic = float(val_acc_logistic/5)
        val_acc_nb = float(val_acc_nb/5) 

        
        test_svm = self.perform_SVM(X_train, Y_train, X_val, Y_val)
        test_logit = self.perform_Logistic(X_train, Y_train, X_val, Y_val)
        test_nb = self.perform_NB(X_train, Y_train, X_val, Y_val)                

        return (val_acc_svm, test_svm), (val_acc_nb, test_nb), (val_acc_logistic, test_logit)
        

    def perform_SVM(self, X_train, Y_train, X_val, Y_val):
        # SVM 
        
        svm_clf = Pipeline([        
            ("linear_svc", LinearSVC(C=1, loss="hinge", max_iter=-1)),
        ])

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
