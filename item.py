class Item: 
    
    def __init__(self, file_id, genre, feature_vector):
        super().__init__()        
        self.file_id = file_id     
        self.genre = genre
        self.feature_vector = feature_vector