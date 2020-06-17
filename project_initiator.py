import os 

from utilities import Utilities
from feature_extractor import FeatureExtractor
from data_point_selector import DataPointSelector
from model_runner import ModelRunner

class Initiator:        

    @classmethod
    def start_processing(cls, path_to_base_folder):
        
        Utilities.prepare_properties_dictionary()
        
        if not os.path.exists(os.path.join(path_to_base_folder, Utilities.get_prop_value(Utilities.BOOK_DESCRIPTOR_KEY))):
            print("Please provide the book descriptor file")
            return

        if not os.path.exists(os.path.join(path_to_base_folder, Utilities.get_prop_value(Utilities.BOOK_REPO_KEY))):
            print("Please provide the book folder")
            return

        if not os.path.exists(os.path.join(os.getcwd(), Utilities.get_prop_value(Utilities.DATA_POINT_KEY))):
            DataPointSelector.select_datapoints(path_to_base_folder)
            return
        else: 
            print("Data Point CSV found in directory, continuing to Feature Extraction")

        if not os.path.exists(os.path.join(os.getcwd(), Utilities.get_prop_value(Utilities.FEATURE_CSV_KEY))):
            extractor = FeatureExtractor(base_folder_address=path_to_base_folder)
            extractor.extract_features()
            return
        else: 
            print("Feature Vector CSV found in directory, continuing to Model Runner")
        
        runner = ModelRunner(path_to_base_folder)
        runner.create_dataset()
        
        
        