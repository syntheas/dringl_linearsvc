import numpy as np
import joblib
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler


def scale_numeric_columns(df):
    # Select numeric and boolean columns
    numeric_cols = df.select_dtypes(include=['number', 'bool']).columns
    
    # Create a copy of the dataframe to avoid modifying the original
    df_scaled = df.copy()
    
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Fit and transform the selected columns
    df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df_scaled


def get_formattedinput(df, vectorizer):
    # separation von numerischen Attributen und Festsetzung auf float Vektor
    df_numeric = df.select_dtypes(include=['number', 'bool'])#.drop(columns= 'change_ausw')
    X_numeric = np.array(df_numeric).astype(float)

    X_texttfidf = vectorizer.transform(df['combined_text'])
    
    # Horizontale Verbindung des Numerik-Vektors und Schlagwort-Vektors
    X = hstack([X_texttfidf, X_numeric])
    
    # Separation der Zielvariable
    y = df['change_auswirkung']
    
    return X, y
        
def get_tfidf_etc(df, vectorizer):
    etc = df.select_dtypes(include=['number', 'bool'])
    tfidf = vectorizer.transform(df['combined_text'])
    return tfidf, etc
        
def get_X(etc, tfidf):
    x_etc = np.array(etc).astype(float)
    x_tfidf = tfidf
    res = hstack([x_tfidf, x_etc])
    return res

def normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val)

def max_proportion(normalized_arr):
    max_val = np.max(normalized_arr)
    sum_val = np.sum(normalized_arr)
    return max_val / sum_val if sum_val != 0 else 0

def get_surence(arr):
    arr = normalize(arr)
    return max_proportion(arr)

def dump_featurenames_dict(df, vectorizer, path):
    # Speicherung von Feature-Namen -> Spaltennamen von numerischen u. Schlagwort-Wörterbuch
    df_numeric = df.select_dtypes(include=['number', 'bool'])
    etc_feature_names = df_numeric.columns.to_list()
    text_keywords = vectorizer.get_feature_names_out().tolist()
    feature_names_dict = {'etc': etc_feature_names, 'text_comb': text_keywords}
    joblib.dump(feature_names_dict, path, compress=('gzip', 3))