import os 
import time
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

from utilities import Utilities 

class ModelRunner: 

    def __init__(self, base_folder_address):
        super().__init__()
        self.base_folder_address = base_folder_address

    def drive_model_runner(self):

        path_to_file = os.path.join(os.getcwd(), Utilities.get_prop_value(Utilities.FEATURE_CSV_KEY))        
        df = pd.read_csv(path_to_file, encoding=Utilities.get_file_encoding(path_to_file=path_to_file))

        strt = time()

        # X_train,X_test,y_train,y_test = train_test_split(df.iloc[:, 1:-1], df.iloc[:, -1],random_state=0)
        # row_list, target = self.convert_data(X_train, y_train)

        # train_data = tf.data.Dataset.from_tensor_slices((row_list, target))
        # train_data = train_data.shuffle(buffer_size=500).batch(50).repeat(200)

        # test_data = tf.data.Dataset.from_tensor_slices((row_list, target))

        # self.run_model(train_data, test_data)

        self.run_multiple_model(df)
        end = time()
        total = end - strt         
        print("Total time : {} minutes".format(total/60))
        
        return

    def convert_data(self, a, b):
        
        row_list =[]     

        for index, rows in a.iterrows(): 
            
            my_list = [rows[i] for i in a.columns]     
            row_list.append(my_list) 
        
        target = [rows for _, rows in b.items()]
            
        return row_list, target
        
    def run_multiple_model(self, df):

        val_acc_svm = 0
        val_acc_logistic = 0
        val_acc_nb = 0

        X = df.iloc[:, 1:-1].to_numpy()
        Y = df.iloc[:, -1].to_numpy()

        X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.20, random_state = 0, shuffle=True, stratify = Y)

        skf = StratifiedKFold(n_splits=5, shuffle=True)

        for train_index, val_index in skf.split(X_train_val, Y_train_val):
            
            X_train, X_val = X_train_val[train_index], X_train_val[val_index]
            Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]
            
            val_acc_svm += perform_SVM(X_train, Y_train, X_val, Y_val)
            val_acc_logistic += perform_Logistic(X_train, Y_train, X_val, Y_val)
            val_acc_nb += perform_NB(X_train, Y_train, X_val, Y_val)            

        val_acc_svm = float(val_acc_svm/5)
        val_acc_logistic = float(val_acc_logistic/5)
        val_acc_nb = float(val_acc_nb/5) 

        print("Average Validation Accuracy")
        print("SVM: {}; Logistic: {}; NB: {}".format(round(val_acc_svm, 4), round(val_acc_logistic, 4), round(val_acc_nb, 4)))

        print("Test Accuracy")
        test_svm = perform_SVM(X_train, Y_train, X_val, Y_val)
        test_logit = perform_Logistic(X_train, Y_train, X_val, Y_val)
        test_nb = perform_NB(X_train, Y_train, X_val, Y_val)
        print("SVM: {}; Logistic: {}; NB: {}".format(round(test_svm, 4), round(test_logit, 4), round(test_nb, 4)))
        

    def perform_SVM(self, X_train, Y_train, X_val, Y_val):
        # SVM 
        
        svm_clf = Pipeline([        
            ("linear_svc", LinearSVC(C=1, loss="hinge")),
        ])

        # print("fitting data to SVM")
        # svm_clf = SVC(gamma='auto')
        svm_clf.fit(X_train, Y_train)
        Y_pred_svm = svm_clf.predict(X_val)
        
        return metrics.accuracy_score(Y_val, Y_pred_svm)

    def perform_Logistic(self, X_train, Y_train, X_val, Y_val):
        
        softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10)
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
