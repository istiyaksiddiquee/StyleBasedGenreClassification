import os 
import time 
import pandas as pd 


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf 
from sklearn.model_selection import train_test_split

from utilities import Utilities 

class ModelRunner: 

    def __init__(self, base_folder_address):
        super().__init__()
        self.base_folder_address = base_folder_address

    def create_dataset(self):

        path_to_file = os.path.join(os.getcwd(), Utilities.get_prop_value(Utilities.FEATURE_CSV_KEY))        
        df = pd.read_csv(path_to_file, encoding=Utilities.get_file_encoding(path_to_file=path_to_file))

        strt = time()

        X_train,X_test,y_train,y_test = train_test_split(df.iloc[:, 1:-1], df.iloc[:, -1],random_state=0)
        row_list, target = self.convert_data(X_train, y_train)

        train_data = tf.data.Dataset.from_tensor_slices((row_list, target))
        train_data = train_data.shuffle(buffer_size=500).batch(50).repeat(200)

        test_data = tf.data.Dataset.from_tensor_slices((row_list, target))

        self.run_model(train_data, test_data)

        end = time()
        total = end - strt         

        pass 

    def convert_data(a, b):
        
        row_list =[]     

        for index, rows in a.iterrows(): 
            
            my_list = [rows[i] for i in a.columns]     
            row_list.append(my_list) 
        
        target = [rows for _, rows in b.items()]
            
        return row_list, target
        
    def run_model(self, train_data, test_data):

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(15, 1)),
            tf.keras.layers.Dense(128,activation='relu'),
            tf.keras.layers.Dense(9, activation='softmax')
        ])

        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(0.001),
            metrics=['accuracy'],
        )

        model.fit(
            train_data,
            epochs=6,
            validation_data=test_data,
        )

        pass 