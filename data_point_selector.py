import os 
import pandas as pd 

from utilities import Utilities 

class DataPointSelector: 

    @classmethod
    def select_datapoints(cls, base_folder_address):
        
        address = os.path.join(base_folder_address, Utilities.get_prop_value(Utilities.BOOK_DESCRIPTOR_KEY))

        df = pd.read_csv(address, sep=";", encoding=Utilities.get_file_encoding(address))

        df.replace(to_replace ="Detective and Mystery", value ="Mystery", inplace=True) 
        df.replace(to_replace ="Sea and Adventure", value ="Adventure", inplace=True) 
        df.replace(to_replace ="Love and Romance", value ="Romance", inplace=True) 
        df.replace(to_replace ="Western Stories", value ="Western", inplace=True) 
        df.replace(to_replace ="Ghost and Horror", value ="Horror", inplace=True) 
        df.replace(to_replace ="Humorous and Wit and Satire", value ="Humorous", inplace=True) 
        df.replace(to_replace ="Christmas Stories", value ="Christmas", inplace=True) 

        literary_15 = df[df.guten_genre == "Literary"].sample(n=15)
        mystery_15 = df[df.guten_genre == "Mystery"].sample(n=15)
        adventure_15 = df[df.guten_genre == "Adventure"].sample(n=15)
        western_15 = df[df.guten_genre == "Western"].sample(n=15)
        romance_15 = df[df.guten_genre == "Romance"].sample(n=15)
        horror_6 = df[df.guten_genre == "Horror"]
        humorous_6 = df[df.guten_genre == "Humorous"]
        christmas_5 = df[df.guten_genre == "Christmas"]
        allegories_2 = df[df.guten_genre == "Allegories"]

        modified_df = literary_15 
        modified_df = modified_df.append(mystery_15)
        modified_df = modified_df.append(adventure_15)
        modified_df = modified_df.append(western_15)
        modified_df = modified_df.append(romance_15)
        modified_df = modified_df.append(horror_6)
        modified_df = modified_df.append(humorous_6)
        modified_df = modified_df.append(christmas_5)
        modified_df = modified_df.append(allegories_2) 

        data = {
            Utilities.BOOK_ID_COLUMN: modified_df[Utilities.BOOK_ID_COLUMN].to_list(), 
            Utilities.GENRE_COLUMN: modified_df[Utilities.GENRE_COLUMN].to_list()
        }
        
        columns = [Utilities.BOOK_ID_COLUMN, Utilities.GENRE_COLUMN]
        
        filtered_df = pd.DataFrame(data, columns=columns, index=None)
        filtered_df.to_csv(Utilities.get_prop_value(Utilities.DATA_POINT_KEY), index=False)

        print("datapoint selection and reporting to csv done.")

        return