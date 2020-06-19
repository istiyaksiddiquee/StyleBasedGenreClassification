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

        literary = df[df.guten_genre == "Literary"].sample(n=30)
        mystery = df[df.guten_genre == "Mystery"].sample(n=30)
        adventure = df[df.guten_genre == "Adventure"].sample(n=30)
        western = df[df.guten_genre == "Western"].sample(n=15)
        romance = df[df.guten_genre == "Romance"].sample(n=15)
        horror = df[df.guten_genre == "Horror"]
        humorous = df[df.guten_genre == "Humorous"]
        christmas = df[df.guten_genre == "Christmas"]
        allegories = df[df.guten_genre == "Allegories"]

        modified_df = literary 
        modified_df = modified_df.append(mystery)
        modified_df = modified_df.append(adventure)
        modified_df = modified_df.append(western)
        modified_df = modified_df.append(romance)
        modified_df = modified_df.append(horror)
        modified_df = modified_df.append(humorous)
        modified_df = modified_df.append(christmas)
        modified_df = modified_df.append(allegories)

        data = {
            Utilities.BOOK_ID_COLUMN: modified_df[Utilities.BOOK_ID_COLUMN].to_list(), 
            Utilities.GENRE_COLUMN: modified_df[Utilities.GENRE_COLUMN].to_list()
        }
        
        columns = [Utilities.BOOK_ID_COLUMN, Utilities.GENRE_COLUMN]
        
        filtered_df = pd.DataFrame(data, columns=columns, index=None)
        filtered_df.to_csv(Utilities.get_prop_value(Utilities.DATA_POINT_KEY), index=False)

        print("datapoint selection and reporting to csv done.")

        return