import pandas as pd 
import numpy as np

class label_encoder(object):
    def __init__(self, sports=[]):
        # unique label, counts in descending order
        self.sports = sports
        self.dmap = {sport:i for i,sport in enumerate(self.sports)}
        self.dmap_inverse = {i:sport for i,sport in enumerate(self.sports)}
        
    def encoder(self, labels):
        return list(map(lambda x:self.dmap[x], labels))

    def decoder(self, codes):
        return list(map(lambda x:self.dmap_inverse[x], codes)) 

if __name__ == '__main__':
    df_raw = pd.read_pickle("./data/df_proper_cleaned.pkl")
    sports= list(df_raw.sport.value_counts(ascending=False).index) 
    code = label_encoder(sports).encoder(['bike','run'])
    decode = label_encoder(sports).decoder(code)
    print(code, decode)
