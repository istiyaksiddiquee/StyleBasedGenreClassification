import re
import nltk 
import string
import chardet
from collections import Counter

class Linguistics: 

    def __init__(self):
        super().__init__()
        self.downtoners_list = ['almost', 'barely', 'hardly', 'merely', 'mildly', 'nearly', 'only', 'partially', 'partly', 'practically', 'scarcely', 'slightly', 'somewhat']

        self.amplifier_list = ['absolutely', 'altogether', 'completely', 'enormously', 'entirely', 'extremely', 'fully', 'greatly', 'highly', 'intensely', 'perfectly', 'strongly', 'thoroughly', 'totally', 'utterly', 'very']

        self.demostrative_list = ['this', 'that', 'these', 'those']

        self.article_list = ['a', 'an', 'the']

    
    def get_sentence_and_word_level_stats(self, input_text):

        char_count = 0
        long_word_count = 0

        punctuation_set = set(string.punctuation)
        punctuation_count = sum([1 for x in input_text if x in punctuation_set])

        sentences = nltk.sent_tokenize(input_text)
        sentence_count = len(sentences)

        tokens = self.parse_text(input_text)
        word_count = len(tokens)

        for word in tokens: 
            char_count += len(word)
            if len(word) > 6: 
                long_word_count += 1        

        noun_count, verb_count, adverb_count, preposition_count, adjective_count, article_count, infinitive_gerund_count, downtoner_count, amplifier_count, demonstrative_count = self.get_grammatical_counts(tokens)

        return sentence_count, word_count, char_count, punctuation_count, long_word_count, noun_count, verb_count, adverb_count, preposition_count, adjective_count, article_count, infinitive_gerund_count, downtoner_count, amplifier_count, demonstrative_count
    
    def get_grammatical_counts(self, tokens):
    
        tags = nltk.pos_tag(tokens=tokens)
        pos_counts = Counter(tag for _, tag in tags)    

        noun_count = 0
        verb_count = 0
        adverb_count = 0
        preposition_count = 0
        adjective_count = 0     
        infinitive_gerund_count = 0
        
        article_count = 0 
        downtoner_count = 0
        amplifier_count = 0 
        demonstrative_count = 0

        for item in tokens:
            if item in self.article_list: 
                article_count += 1
            if item in self.downtoners_list: 
                downtoner_count += 1 
            if item in self.amplifier_list: 
                amplifier_count += 1 
            if item in self.demostrative_list: 
                demonstrative_count += 1        

        if pos_counts.get('NN') != None: 
            noun_count += pos_counts.get('NN')
        if pos_counts.get('NNS') != None: 
            noun_count += pos_counts.get('NNS')
        if pos_counts.get('NNP') != None: 
            noun_count += pos_counts.get('NNP')
        if pos_counts.get('NNPS') != None: 
            noun_count += pos_counts.get('NNPS')    
        
        if pos_counts.get('VB') != None: 
            verb_count += pos_counts.get('VB')    
        if pos_counts.get('VBD') != None: 
            verb_count += pos_counts.get('VBD')    
        if pos_counts.get('VBG') != None: 
            verb_count += pos_counts.get('VBG')
            infinitive_gerund_count += pos_counts.get('VBG')
        if pos_counts.get('VBN') != None: 
            verb_count += pos_counts.get('VBN')    
        if pos_counts.get('VBP') != None: 
            verb_count += pos_counts.get('VBP')    
        if pos_counts.get('VBZ') != None: 
            verb_count += pos_counts.get('VBZ')    

        if pos_counts.get('RB') != None: 
            adverb_count += pos_counts.get('RB')    
        if pos_counts.get('RBR') != None: 
            adverb_count += pos_counts.get('RBR')    
        if pos_counts.get('RBS') != None: 
            adverb_count += pos_counts.get('RBS') 

        if pos_counts.get('IN') != None: 
            preposition_count += pos_counts.get('IN') 

        if pos_counts.get('JJ') != None: 
            adjective_count += pos_counts.get('JJ')    
        if pos_counts.get('JJR') != None: 
            adjective_count += pos_counts.get('JJR')    
        if pos_counts.get('JJS') != None: 
            adjective_count += pos_counts.get('JJS') 
        
        return (noun_count, verb_count, adverb_count, preposition_count, adjective_count, article_count, infinitive_gerund_count, downtoner_count, amplifier_count, demonstrative_count)

    def parse_text(self, input_text): 

        regex = r"\w+(?:[-'’]\w+)*|'|[-.(]+|\S\w*"
        parsed_strings = re.findall(regex, input_text)
        return_list = [re.sub('[^A-Za-z0-9]+', '', item) for item in parsed_strings if len(re.sub('[^A-Za-z0-9]+', '', item)) != 0 and item not in string.punctuation+''.join(['’', '”', '“', '”'])   and not item.isnumeric()]
        return return_list