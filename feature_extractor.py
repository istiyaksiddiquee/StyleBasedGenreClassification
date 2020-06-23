import os 
import sys
import glob 
import codecs
import chardet 
import pandas as pd
from time import time

from concurrent.futures import ThreadPoolExecutor, as_completed, wait

from item import Item
from utilities import Utilities
from linguistic_analysis import Linguistics

class FeatureExtractor: 

    def __init__(self, base_folder_address):
        super().__init__()
        self.base_folder_address = base_folder_address

    def extract_features(self):

        path_to_file = os.path.join(os.getcwd(), Utilities.get_prop_value(Utilities.DATA_POINT_KEY))
        
        df = pd.read_csv(path_to_file, encoding=Utilities.get_file_encoding(path_to_file=path_to_file))
        book_id_list = df[Utilities.BOOK_ID_COLUMN].tolist()
        genre_list = df[Utilities.GENRE_COLUMN].tolist()

        strt = time()
        info_depot = self.file_driver(book_id_list, genre_list)
        end = time()

        total = end - strt 
        
        print("Total time : {} minutes".format(total/60))
        
        lines = []

        for item in info_depot:
            file_info = info_depot.get(item)
            line = [file_info.file_id]
            
            for f in file_info.feature_vector: 
                line.append(f)
                
            line.append(file_info.genre)
            
            lines.append(line)
                
        columns = ['book_id', 'avg_sentence_length', 'avg_char_per_word', 'avg_punctuation_per_sentence', 'long_word_ratio', 'noun_ratio', 
                'verb_ratio', 'adverb_ratio', 'preposition_ratio', 'adjective_ratio', 'article_ratio', 'gerund_infinitive_ratio', 
                'downtoner_ratio', 'amplifier_ratio', 'demonstrative_ratio', 'type_token_ratio', 'genre']

        df = pd.DataFrame(lines, columns=columns, index=None)
        df.to_csv(Utilities.get_prop_value(Utilities.FEATURE_CSV_KEY), index=False)
        
        print("writing features to file done.")
        return
    
    def file_driver(self, book_id_list, genre_list):

        info_depot = {}
        futures = []        
                
        pool = ThreadPoolExecutor(Utilities.FILE_POOL_SIZE)

        k = 0
        steps = len(book_id_list)
        
        for i in range(steps):

            l = 10
            if k != 0: 
                i =  k + 1
                if i >= steps: 
                    break

            if i+l >= steps: 
                l = steps-i
            
            for k in range(i, i+l):
                
                futures.append(pool.submit(self.file_task, (book_id_list[k], genre_list[k])))
            
            wait(futures)

            for f in as_completed(futures):

                item = f.result()
                info_depot[item.file_id] = item
                print("file_id: {}, genre: {}, feature_vector: {}".format(item.file_id, item.genre, item.feature_vector))
        
        return info_depot

    def file_task(self, info): 

        file_name, genre = info
        new_book_name_prefix = file_name.split(sep=".")[0]
        
        path_to_glob = os.path.join(self.base_folder_address, Utilities.get_prop_value(Utilities.BOOK_REPO_KEY), new_book_name_prefix+"*.html")
        path_to_file = glob.glob(path_to_glob)[0]
        print(path_to_file)
        with codecs.open(path_to_file, "r", encoding='utf8') as file:
            unformatted_text = file.read()        

        p_removed_text = unformatted_text.replace("<p>", "")
        feature_vector = self.extract_feature_vector(p_removed_text)
        return Item(file_name, genre, feature_vector=feature_vector)            

    def extract_feature_vector(self, p_removed_text):

        analysis = Linguistics()
        
        futures = []
        pool = ThreadPoolExecutor(Utilities.CHUNK_POOL_SIZE)
        
        total_sentence_count = 0 
        total_word_count = 0 
        total_char_count = 0 
        total_punctuation_count = 0
        total_long_word_count = 0 
        total_noun_count = 0 
        total_verb_count = 0 
        total_adverb_count = 0 
        total_preposition_count = 0        
        total_adjective_count = 0 
        total_article_count = 0 
        total_infinitive_gerund_count = 0 
        total_downtoner_count = 0
        total_amplifier_count = 0 
        total_demonstrative_count = 0

        total_unique_word = len(set(analysis.parse_text(p_removed_text)))

        steps = len(p_removed_text)/1000
        if (steps - int(steps)) != 0:
            steps = int(steps) + 1
        
        j = 0
        for i in range(steps):

            l = 10 
            if j != 0: 
                i =  j + 1
                if i >= steps: 
                    break

            if i+l >= steps: 
                l = steps-i
            
            for j in range(i, i+l):
                start_index = j*1000 
                end_index = (j+1)*1000
                futures.append(pool.submit(analysis.get_sentence_and_word_level_stats, p_removed_text[start_index:end_index]))
            
            wait(futures)

            for f in as_completed(futures):
                sentence_count, word_count, char_count, punctuation_count, long_word_count, noun_count, verb_count, adverb_count, preposition_count, adjective_count, article_count, infinitive_gerund_count, downtoner_count, amplifier_count, demonstrative_count = f.result() 
                total_sentence_count += sentence_count
                total_word_count += word_count
                total_char_count += char_count
                total_punctuation_count += punctuation_count
                total_long_word_count += long_word_count
                total_noun_count += noun_count
                total_verb_count += verb_count
                total_adverb_count += adverb_count
                total_preposition_count += preposition_count
                total_adjective_count += adjective_count
                total_article_count += article_count
                total_infinitive_gerund_count += infinitive_gerund_count
                total_downtoner_count += downtoner_count
                total_amplifier_count += amplifier_count
                total_demonstrative_count += demonstrative_count


        avg_sentence_length = round(float(total_sentence_count/total_word_count), 4)
        avg_char_per_word = round(float(total_char_count/total_word_count), 4)
        avg_punctuation_per_sentence = round(float(total_punctuation_count/total_sentence_count), 4)
        long_word_ratio = round(float(total_long_word_count/total_word_count), 4)
        noun_ratio = round(float(total_noun_count/total_word_count), 4)
        verb_ratio = round(float(total_verb_count/total_word_count), 4)
        adverb_ratio = round(float(total_adverb_count/total_word_count), 4)
        preposition_ratio = round(float(total_preposition_count/total_word_count), 4)
        adjective_ratio = round(float(total_adjective_count/total_word_count), 4)
        article_ratio = round(float(total_article_count/total_word_count), 4)
        gerund_infinitive_ratio = round(float(total_infinitive_gerund_count/total_word_count), 4)
        downtoner_ratio = round(float(total_downtoner_count/total_word_count), 4)
        amplifier_ratio = round(float(total_amplifier_count/total_word_count), 4)
        demonstrative_ratio = round(float(total_demonstrative_count/total_word_count), 4)
        type_token_ratio = round(float(total_unique_word/total_word_count), 4)


        return (avg_sentence_length, avg_char_per_word, avg_punctuation_per_sentence, long_word_ratio, noun_ratio, 
                verb_ratio, adverb_ratio, preposition_ratio, adjective_ratio, article_ratio, gerund_infinitive_ratio, 
                downtoner_ratio, amplifier_ratio, demonstrative_ratio, type_token_ratio)
