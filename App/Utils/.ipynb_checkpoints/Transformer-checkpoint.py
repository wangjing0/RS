import pandas as pd 
import numpy as np

class Transformer(object):
    def __init__(self, fillnan=True):
        self._NAME_ = 'Transformer'
        
        # categorical features to dummify and then drop
        self.feature_categorical = ['userId', 'gender','sport']
        
        # numeric features to keep
        self.feature_numeric = ['heart_rate_mean', 'heart_rate_std', 
                                   'altitude_mean', 'altitude_std',
                                   'ascend_m', 'descend_m', 'distance_total_m',
                                   'speed_mean','speed_std','duration_s','calories']
        
        # from the top 20 sport list
        self.sports = ['bike', 'run', 'mountain bike', 'bike (transport)', 'indoor cycling',
                       'walk', 'orienteering', 'cross-country skiing',
                       'core stability training', 'fitness walking', 'skate', 'roller skiing',
                       'hiking', 'circuit training', 'kayaking', 'rowing', 'weight training',
                       'soccer', 'downhill skiing', 'gymnastics']
        
        self.gender = ['male','female','unknown']
        
        self.FILLNAN = fillnan
        
    def fit(self, X, y=None):
        if self.FILLNAN:
            col_list = list(set(X.columns) & set(self.feature_numeric))
            self.mean = X[col_list].mean()
        
    def transform(self, X, y=None):
        features = X.columns
        df = pd.DataFrame()
        for feature in features:
            if feature =='start_time':
                date = pd.to_datetime(X[feature])
                df['year'] = date.dt.year
                df['month'] = date.dt.month
                df['weekday'] = date.dt.weekday
            if feature =='gender':
                for g in self.gender:
                    df[g] = (X[feature]==g).astype(int)
                    
            if feature == 'sport': # dummy transform onto fixed categorical labels
                for s in self.sports:
                    df[s] = (X[feature]==s).astype(int)
            
            if feature in self.feature_numeric:
                df[feature] = X[feature]
        
        if hasattr(self, 'mean') and self.FILLNAN:
            df = df.fillna(self.mean)
            
        # engineering new features
        #if {'calories', 'duration_s'}.issubset(df.columns):
        #    df['calories_rate'] = (1000*df['calories'])/df['duration_s']
        
        return df
        
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    
class label_encoder(object):
    def __init__(self, df=None):
        # unique label, counts in descending order
        self.__NAME__ = 'Sport_LabelEncoder'
        self.sports = list(df.sport.value_counts(ascending=False).index) 
        self.dmap = {sport:i for i,sport in enumerate(self.sports)}
        self.dmap_inverse = {i:sport for i,sport in enumerate(self.sports)}
        
    def encoder(self, labels):
        return list(map(lambda x:self.dmap[x], labels))

    def decoder(self, codes):
        return list(map(lambda x:self.dmap_inverse[x], codes)) 

if __name__ == '__main__':
    df_raw = pd.read_pickle("./data/df_proper_cleaned.pkl")
    df_transformed = Transformer(fillnan=True).fit_transform(df_raw)
    print(df_transformed.info())
    df_transformed.head()