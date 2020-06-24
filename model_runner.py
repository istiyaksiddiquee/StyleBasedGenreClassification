import os 
import operator
import numpy as np
import pandas as pd 
from time import time
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn  import feature_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

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
        self.run_multiple_model(X, Y)
        multi_model_end_time = time()

        nn_start_time = time()
        nn_acc = self.run_nn(X, Y)
        nn_end_time = time()        

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
    
    def get_best_params(self, method, X, Y):

        param_dict = {}
        if method == "SVM_POLY": 

            method=SVC(kernel='poly')
            param_dict={
                'C': [0.01, 0.1, 1, 10, 100, 1000],
                'degree': [1, 2, 3, 4, 5, 6, 7],
                'coef0': [ 0.05, 0.1, 0.5, 1, 5],
                'tol': [0.001, 0.005, 0.01, 0.05]
            }
        elif method == "SVM_RBF": 
            method=SVC(kernel='rbf')
            param_dict={
                'C': [0.01, 0.1, 1, 10, 100, 1000],
                'degree': [1, 2, 3, 4, 5, 6, 7],
                'coef0': [ 0.05, 0.1, 0.5, 1, 5],
                'tol': [0.001, 0.005, 0.01, 0.05]
            }
        elif method == 'MNB':
            method = MultinomialNB()
            param_dict = { 'alpha': [1000, 100, 10, 5, 1, 0.5, 0.1, 0, 0.05, 0.01, 0.001]}
        elif method == "LR": 
            method=LogisticRegression()
            param_dict={
                'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5],
                'tol': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5],
                'intercept_scaling': [0.005, 0.01, 0.05, 0.1],
                'multi_class': ['multinomial'],
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], 
                'class_weight': ['balanced'], 
                'max_iter': [10000]
            }
        elif method == 'RF':
            method=RandomForestClassifier()
            param_dict={
                'criterion': ['gini', 'entropy'], 
                'max_depth': [10, 20, 50, 100],
                'max_leaf_nodes': [10, 20, 50, 100],
                'max_features': ['auto']
            }        

        return self.calculate_best_param(method, param_dict, X, Y)

    def run_multiple_model(self, X, Y):

        n_split = 3
        
        val_acc_svm_poly = 0
        val_acc_svm_rbf = 0
        val_acc_logistic = 0
        val_acc_nb = 0
        val_acc_rf = 0
        val_acc_ensemble = 0

        val_precision_svm_poly = 0
        val_precision_svm_rbf = 0
        val_precision_logistic = 0
        val_precision_nb = 0
        val_precision_rf = 0
        val_precision_ensemble = 0

        val_recall_svm_poly = 0
        val_recall_svm_rbf = 0
        val_recall_logistic = 0
        val_recall_nb = 0
        val_recall_rf = 0
        val_recall_ensemble = 0
        
        val_f1_svm_poly = 0
        val_f1_svm_rbf = 0
        val_f1_logistic = 0
        val_f1_nb = 0
        val_f1_rf = 0
        val_f1_ensemble = 0


        X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.25, random_state = 0, shuffle=True, stratify = Y)

        # selecting 20 best features
        X_train_val, X_test = self.select_k_features(k=20, X_train=X_train_val, X_test=X_test, Y_train=Y_train_val)

        self.bias_variance_decomp(X_train_val, X_test, Y_train_val, Y_test)

        # selecting best values for classifiers
        svm_start = time()
        print("Retrieving best param for Polynomial Kernel SVM")
        poly_svm_best_params = self.get_best_params("SVM_POLY", X_train_val, Y_train_val)
        svm_end = time()
        print("Time spent: {}".format(float((svm_end-svm_start)/60)))

        rbf_start = time()
        print("Retrieving best param for RBF Kernel SVM")
        rbf_svm_best_params = self.get_best_params("SVM_RBF", X_train_val, Y_train_val)
        rbf_end = time()
        print("Time spent: {}".format(float((rbf_end-rbf_start)/60)))
        
        nb_start = time()
        print("Retrieving best param for MultiNomial Naive Bayes")
        nb_best_params = self.get_best_params("MNB", X_train_val, Y_train_val)
        nb_end = time()
        print("Time spent: {}".format(float((nb_end-nb_start)/60)))

        lr_start = time()
        print("Retrieving best param for Logistic Regression")
        lr_best_params = self.get_best_params("LR", X_train_val, Y_train_val)
        lr_end = time()
        print("Time spent: {}".format(float((lr_end-lr_start)/60)))

        rf_start = time()
        print("Retrieving best param for Random Forest")
        rf_best_params = self.get_best_params("RF", X_train_val, Y_train_val)
        rf_end = time()
        print("Time spent: {}".format(float((rf_end-rf_start)/60)))

        skf = StratifiedKFold(n_splits=n_split, shuffle=True)

        for train_index, val_index in skf.split(X_train_val, Y_train_val):
            
            X_train, X_val = X_train_val[train_index], X_train_val[val_index]
            Y_train, Y_val = Y_train_val[train_index], Y_train_val[val_index]

            X_train, Y_train = self.tackle_data_imbalance(X_train, Y_train)

            acc_svm_poly, precision_svm_poly, recall_svm_poly, f1_score_svm_poly = self.perform_SVM_with_Polynomial_kernel(poly_svm_best_params, X_train, Y_train, X_val, Y_val)
            acc_svm_rbf, precision_svm_rbf, recall_svm_rbf, f1_score_svm_rbf = self.perform_SVM_with_RBF_kernel(rbf_svm_best_params, X_train, Y_train, X_val, Y_val)
            acc_lr, precision_lr, recall_lr, f1_score_lr = self.perform_Logistic(lr_best_params, X_train, Y_train, X_val, Y_val)
            acc_nb, precision_nb, recall_nb, f1_score_nb = self.perform_NB(nb_best_params, X_train, Y_train, X_val, Y_val)    
            acc_rf, precision_rf, recall_rf, f1_score_rf = self.perform_random_forest(rf_best_params, X_train, Y_train, X_val, Y_val)
            acc_en, precision_en, recall_en, f1_score_en = self.do_ensemble_and_learn(rf_best_params, lr_best_params, poly_svm_best_params, X_train, Y_train, X_val, Y_val)

            val_acc_svm_poly += acc_svm_poly
            val_acc_svm_rbf += acc_svm_rbf
            val_acc_logistic += acc_lr
            val_acc_nb += acc_nb
            val_acc_rf += acc_rf
            val_acc_ensemble += acc_en

            val_precision_svm_poly += precision_svm_poly
            val_precision_svm_rbf += precision_svm_rbf
            val_precision_logistic += precision_lr
            val_precision_nb += precision_nb
            val_precision_rf += precision_rf
            val_precision_ensemble += precision_en

            val_recall_svm_poly += recall_svm_poly
            val_recall_svm_rbf += recall_svm_rbf
            val_recall_logistic += recall_lr
            val_recall_nb += recall_nb
            val_recall_rf += recall_rf
            val_recall_ensemble += recall_en

            val_f1_svm_poly += f1_score_svm_poly
            val_f1_svm_rbf += f1_score_svm_rbf
            val_f1_logistic += f1_score_lr
            val_f1_nb += f1_score_nb
            val_f1_rf += f1_score_rf
            val_f1_ensemble += f1_score_en
            
        val_acc_svm_poly = float(val_acc_svm_poly/n_split)
        val_acc_svm_rbf = float(val_acc_svm_rbf/n_split)
        val_acc_logistic = float(val_acc_logistic/n_split)
        val_acc_nb = float(val_acc_nb/n_split) 
        val_acc_rf = float(val_acc_rf/n_split) 
        val_acc_ensemble = float(val_acc_ensemble/n_split) 

        val_precision_svm_poly = float(val_precision_svm_poly/n_split)
        val_precision_svm_rbf = float(val_precision_svm_rbf/n_split)
        val_precision_logistic = float(val_precision_logistic/n_split)
        val_precision_nb = float(val_precision_nb/n_split)
        val_precision_rf = float(val_precision_rf/n_split)
        val_precision_ensemble = float(val_precision_ensemble/n_split)

        val_recall_svm_poly = float(val_recall_svm_poly/n_split)
        val_recall_svm_rbf = float(val_recall_svm_rbf/n_split)
        val_recall_logistic = float(val_recall_logistic/n_split)
        val_recall_nb = float(val_recall_nb/n_split)
        val_recall_rf = float(val_recall_rf/n_split)
        val_recall_ensemble = float(val_recall_ensemble/n_split)
        
        val_f1_svm_poly = float(val_f1_svm_poly/n_split)
        val_f1_svm_rbf = float(val_f1_svm_rbf/n_split)
        val_f1_logistic = float(val_f1_logistic/n_split)
        val_f1_nb = float(val_f1_nb/n_split)
        val_f1_rf = float(val_f1_rf/n_split)
        val_f1_ensemble = float(val_f1_ensemble/n_split)

        test_svm_poly_acc, test_svm_poly_precision, test_svm_poly_recall, test_svm_poly_f1 = self.perform_SVM_with_Polynomial_kernel(poly_svm_best_params, X_train, Y_train, X_test, Y_test)
        test_svm_rbf_acc, test_svm_rbf_precision, test_svm_rbf_recall, test_svm_rbf_f1 = self.perform_SVM_with_RBF_kernel(rbf_svm_best_params, X_train, Y_train, X_test, Y_test)
        test_logit_acc, test_logit_precision, test_logit_recall, test_logit_f1 = self.perform_Logistic(lr_best_params, X_train, Y_train, X_test, Y_test)
        test_nb_acc, test_nb_precision,test_nb_recall, test_nb_f1 = self.perform_NB(nb_best_params, X_train, Y_train, X_test, Y_test)
        test_rf_acc, test_rf_precision, test_rf_recall, test_rf_f1 = self.perform_random_forest(rf_best_params, X_train, Y_train, X_test, Y_test)
        test_ensemble_acc, test_ensemble_precision, test_ensemble_recall, test_ensemble_f1 = self.do_ensemble_and_learn(rf_best_params, lr_best_params, poly_svm_best_params, X_train, Y_train, X_test, Y_test)

        print("Performance Report")
        print("--------------------------------------")
        print("SVM With PolyKernel Report")
        print("validation accuracy: {}, test accuracy: {}".format(val_acc_svm_poly, test_svm_poly_acc))
        print("validation precision: {}, test precision: {}".format(val_precision_svm_poly, test_svm_poly_precision))
        print("validation recall: {}, test recall: {}".format(val_recall_svm_poly, test_svm_poly_recall))
        print("validation f1_score: {}, test f1_score: {}".format(val_f1_svm_poly, test_svm_poly_f1))
        
        print("SVM With RBF-Kernel Report")
        print("validation accuracy: {}, test accuracy: {}".format(val_acc_svm_rbf, test_svm_rbf_acc))
        print("validation precision: {}, test precision: {}".format(val_precision_svm_rbf, test_svm_rbf_precision))
        print("validation recall: {}, test recall: {}".format(val_recall_svm_rbf, test_svm_rbf_recall))
        print("validation f1_score: {}, test f1_score: {}".format(val_f1_svm_rbf, test_svm_rbf_f1))
        
        print("Multi-NB Report")
        print("validation accuracy: {}, test accuracy: {}".format(val_acc_nb, test_nb_acc))
        print("validation precision: {}, test precision: {}".format(val_precision_nb, test_nb_precision))
        print("validation recall: {}, test recall: {}".format(val_recall_nb, test_nb_recall))
        print("validation f1_score: {}, test f1_score: {}".format(val_f1_nb, test_nb_f1))
        
        print("Multi-Logistic Report")
        print("validation accuracy: {}, test accuracy: {}".format(val_acc_logistic, test_logit_acc))
        print("validation precision: {}, test precision: {}".format(val_precision_logistic, test_logit_precision))
        print("validation recall: {}, test recall: {}".format(val_recall_logistic, test_logit_recall))
        print("validation f1_score: {}, test f1_score: {}".format(val_f1_logistic, test_logit_f1))

        print("Random-Forrest Report")
        print("validation accuracy: {}, test accuracy: {}".format(val_acc_rf, test_rf_acc))
        print("validation precision: {}, test precision: {}".format(val_precision_rf, test_rf_precision))
        print("validation recall: {}, test recall: {}".format(val_recall_rf, test_rf_recall))
        print("validation f1_score: {}, test f1_score: {}".format(val_f1_rf, test_rf_f1))

        print("Ensemble Report")
        print("validation accuracy: {}, test accuracy: {}".format(val_acc_ensemble, test_ensemble_acc))
        print("validation precision: {}, test precision: {}".format(val_precision_ensemble, test_ensemble_precision))
        print("validation recall: {}, test recall: {}".format(val_recall_ensemble, test_ensemble_recall))
        print("validation f1_score: {}, test f1_score: {}".format(val_f1_ensemble, test_ensemble_f1))
        
        return         

    def perform_SVM_with_Polynomial_kernel(self, method_params, X_train, Y_train, X_test, Y_test):
        
        svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(
                    kernel="poly", 
                    degree=method_params.get('degree'), 
                    coef0=method_params.get('coef0'), 
                    C=method_params.get('C'), 
                    tol=method_params.get('tol')
                    )
            )
        ])
        
        svm_clf.fit(X_train, Y_train)
        Y_pred = svm_clf.predict(X_test)
        
        accuracy = metrics.accuracy_score(Y_test, Y_pred)
        precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(Y_test, Y_pred, beta=1.0, average='macro')

        return (accuracy, precision, recall, f1_score)

    
    def perform_SVM_with_RBF_kernel(self, method_params, X_train, Y_train, X_test, Y_test):
        
        svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(
                        kernel="rbf", 
                        degree=method_params.get('degree'), 
                        coef0=method_params.get('coef0'), 
                        C=method_params.get('C'), 
                        tol=method_params.get('tol')
                    )   
            )
        ])
        
        svm_clf.fit(X_train, Y_train)
        
        Y_pred = svm_clf.predict(X_test)
        
        accuracy = metrics.accuracy_score(Y_test, Y_pred)
        precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(Y_test, Y_pred, beta=1.0, average='macro')

        return (accuracy, precision, recall, f1_score)


    def perform_Logistic(self, method_params, X_train, Y_train, X_test, Y_test):
                
        clf = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                                multi_class="multinomial", 
                                C=method_params.get('C'), 
                                tol=method_params.get('tol'),
                                intercept_scaling=method_params.get('intercept_scaling'),
                                solver=method_params.get('solver'),
                                class_weight='balanced', 
                                max_iter=10000,
                                n_jobs=-1
                        )
                )
            ])        
                        
        clf.fit(X_train, Y_train)
        
        Y_pred = clf.predict(X_test)
        
        accuracy = metrics.accuracy_score(Y_test, Y_pred)
        precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(Y_test, Y_pred, beta=1.0, average='macro')

        return (accuracy, precision, recall, f1_score)

    def perform_NB(self, method_params, X_train, Y_train, X_test, Y_test):

        nb = MultinomialNB(alpha=method_params.get('alpha'))

        nb.fit(X_train, Y_train)

        Y_pred = nb.predict(X_test)

        accuracy = metrics.accuracy_score(Y_test, Y_pred)
        precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(Y_test, Y_pred, beta=1.0, average='macro')

        return (accuracy, precision, recall, f1_score)

    def perform_random_forest(self, method_params, X_train, Y_train, X_test, Y_test):

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                        criterion=method_params.get('criterion'),
                        max_depth=method_params.get('max_depth'), 
                        max_leaf_nodes=method_params.get('max_leaf_nodes'), 
                        max_features=method_params.get('max_features')
                    )
            )
        ])
        
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)

        accuracy = metrics.accuracy_score(Y_test, Y_pred)
        precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(Y_test, Y_pred, beta=1.0, average='macro')

        return (accuracy, precision, recall, f1_score)

    def do_ensemble_and_learn(self, rf_method_params, lr_method_params, svm_method_params, X_train, Y_train, X_test, Y_test): 

        clf1 = ("svm_clf", SVC(
                            kernel="poly", 
                            degree=svm_method_params.get('degree'), 
                            coef0=svm_method_params.get('coef0'), 
                            C=svm_method_params.get('C'), 
                            tol=svm_method_params.get('tol')
                        )
                )
        clf2 = ("lr", LogisticRegression(
                                multi_class="multinomial", 
                                C=lr_method_params.get('C'), 
                                tol=lr_method_params.get('tol'),
                                intercept_scaling=lr_method_params.get('intercept_scaling'),
                                solver=lr_method_params.get('solver'),
                                class_weight='balanced', 
                                max_iter=10000,
                                n_jobs=-1
                        )
                )
        clf3 = ("rf", RandomForestClassifier(
                        criterion=rf_method_params.get('criterion'),
                        max_depth=rf_method_params.get('max_depth'), 
                        max_leaf_nodes=rf_method_params.get('max_leaf_nodes'), 
                        max_features=rf_method_params.get('max_features')
                    )
                )       
        
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

        accuracy = metrics.accuracy_score(Y_test, Y_preds)
        precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(Y_test, Y_preds, beta=1.0, average='macro')

        return (accuracy, precision, recall, f1_score)

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

    def calculate_best_param(self, method, param, X, Y):

        gsc = GridSearchCV(
                estimator=method,
                param_grid=param,
                cv=5, 
                scoring='accuracy', 
                verbose=0, 
                n_jobs=-1
            )
        
        grid_result = gsc.fit(X, Y)

        return grid_result.best_params_