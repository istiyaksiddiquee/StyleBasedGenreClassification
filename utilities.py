import os 
import chardet

class Utilities: 

    KEY_VALUE_STORE = {}
    
    FILE_POOL_SIZE = 10
    CHUNK_POOL_SIZE = 10

    PROPERTIES_FILE_NAME = "properties.config"
    BOOK_REPO_KEY = "BOOK_REPO_FOLDER"
    BOOK_DESCRIPTOR_KEY = "BOOK_DESCRIPTOR_CSV"
    FEATURE_CSV_KEY = "FEATURE_CSV"
    DATA_POINT_KEY = "DATA_POINT_CSV"
    BOOK_ID_COLUMN = "book_id"
    GENRE_COLUMN = "guten_genre"

    @classmethod
    def get_file_encoding(cls, path_to_file):
        with open(path_to_file, 'rb') as rawdata:
            result = chardet.detect(rawdata.read(100000))
        return result.get('encoding')
    
    @classmethod
    def get_prop_value(cls, key):
        value = Utilities.KEY_VALUE_STORE.get(key)
        if value == None: 
            print("Requested value not found")
            return None
        else: 
            return value

    @classmethod
    def prepare_properties_dictionary(cls):
        path_to_prop_file = os.path.join(os.getcwd(), Utilities.PROPERTIES_FILE_NAME)
        encoding = Utilities.get_file_encoding(path_to_file=path_to_prop_file)
        file = open(path_to_prop_file, "r", encoding=encoding)
        lines = file.readlines()        
        
        for line in lines:
            parts = line.rstrip('\n').split(sep="=")
            Utilities.KEY_VALUE_STORE[parts[0]] = parts[1]