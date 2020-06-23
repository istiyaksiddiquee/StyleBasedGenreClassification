import os 
import subprocess
from time import time

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
            data_start = time()
            DataPointSelector.select_datapoints(path_to_base_folder)
            data_end = time()

            print("Data Selection took : {} minutes".format((data_end-data_start)/60))
        else: 
            print("Data Point CSV found in directory, continuing to Feature Extraction")

        if not os.path.exists(os.path.join(os.getcwd(), Utilities.get_prop_value(Utilities.PYTHON_FEATURE_CSV))):

            py_start = time()
            extractor = FeatureExtractor(base_folder_address=path_to_base_folder)
            extractor.extract_features()
            py_end = time()

            print("Python Extractor took : {} minutes".format((py_end-py_start)/60))
        else: 
            print("Python Feature Vector CSV found in directory, continuing to run Java project")
        
        if not os.path.exists(os.path.join(os.getcwd(), Utilities.get_prop_value(Utilities.JAVA_FEATURE_CSV))):
            
            bat_file_name = r'command.bat'
            folder_path = os.path.join(path_to_base_folder, Utilities.get_prop_value(Utilities.BOOK_REPO_FOLDER))
            output_file_name = ".\\" + Utilities.get_prop_value(Utilities.JAVA_FEATURE_CSV)
            book_descriptor_file_name = ".\\" + Utilities.get_prop_value(Utilities.BOOK_DESCRIPTOR_CSV)
            data_points_file_name = ".\\" + Utilities.get_prop_value(Utilities.DATA_POINT_CSV)
            
            java_start = time()
            x = subprocess.call([bat_file_name, folder_path, output_file_name, book_descriptor_file_name, data_points_file_name])
            java_end = time()

            print("Java Project took : {} minutes".format((java_end-java_start)/60))
            
        else: 
            print("Java output Feature Vector CSV found in directory, continuing to Model Runner")
        
        runner = ModelRunner(path_to_base_folder)
        runner.drive_model_runner()