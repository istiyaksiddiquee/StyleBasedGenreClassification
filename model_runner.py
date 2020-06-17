import os 
import time 
import pandas as pd 

from utilities import Utilities 

class ModelRunner: 

    def __init__(self, base_folder_address):
        super().__init__()
        self.base_folder_address = base_folder_address

    def create_dataset(self):

        path_to_file = os.path.join(os.getcwd(), Utilities.get_prop_value(Utilities.FEATURE_CSV_KEY))
        
        df = pd.read_csv(path_to_file, encoding=Utilities.get_file_encoding(path_to_file=path_to_file))
        book_id_list = df[Utilities.BOOK_ID_COLUMN].tolist()
        genre_list = df[Utilities.GENRE_COLUMN].tolist()

        strt = time()
        info_depot = self.file_driver(book_id_list, genre_list)
        end = time()

        total = end - strt 
        

        pass 