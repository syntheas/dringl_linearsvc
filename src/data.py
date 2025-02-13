from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import numpy as np
import dill
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import h5py
import scipy



# Get the directory of the executing Python file
script_dir = Path(__file__).parent.resolve()
data_folder= Path(script_dir / "../input")
outputdependence_folder= Path(script_dir / "../output_dependence")

datatrain_path = str(data_folder / "features4dringl4linearsvc_train.csv")
datatrainsampled_path = str(data_folder / "features4dringl4linearsvc_trainsampled.h5")
datatest_path = str(data_folder / "features4dringl4linearsvc_test.csv")

output_dependence_path = str(outputdependence_folder / "dependence_{}.pkl")



def get_data():
    df_train = load_df(datatrain_path)
    df_testval = load_df(datatest_path)
    # df = filter_pre2(df)
    
    # Aufteilung in val u. Testdaten, dabei wird auf Gleichverteilung des auswichkeitsattributs geachtet
    df_val, _ = train_test_split(df_testval, test_size=0.57, random_state=42, stratify=df_testval['urgency'])
    
    X_train, y_train, vectorizer, scaler, selector = get_x_y(df_train)
    X_val, y_val, _, _, _ = get_x_y(df_val, vectorizer, scaler, selector)
    
    return X_train, y_train, X_val, y_val


def get_sampled_data():
    X_train, y_train = load_h5_sparse(datatrainsampled_path)
    X_train, scaler = scale_sampled_data(X_train)
    df_testval = load_df(datatest_path)
    # Aufteilung in val u. Testdaten, dabei wird auf Gleichverteilung des auswichkeitsattributs geachtet
    df_val, _ = train_test_split(df_testval, test_size=0.57, random_state=42, stratify=df_testval['urgency'])
    vectorizer, selector = load_dependence()
    X_val, y_val, _, _, _ = get_x_y(df_val, vectorizer, scaler, selector)

    return X_train, y_train, X_val, y_val

def scale_sampled_data(X_train):
    # Convert sparse to dense if needed
    if scipy.sparse.issparse(X_train):
        X_train = X_train.toarray()
    
    # Split the data
    data_to_scale = X_train[:,:-5000]
    data_not_to_scale = X_train[:,-5000:]
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_to_scale)
    if not scipy.sparse.issparse(scaled_data):
        # Convert to sparse matrix if it's not already
        scaled_data = scipy.sparse.csr_matrix(scaled_data)
    X_train_combined = scipy.sparse.hstack([scaled_data, data_not_to_scale])
    return X_train_combined, scaler

def load_df(path):
    df = pd.read_csv(path, dtype={'urgency': str})
    df.drop(columns="id", inplace=True)
    
    return df
    
def load_h5_sparse(filepath):
    """Loads a sparse matrix and y from an HDF5 file."""
    with h5py.File(filepath, "r") as f:
        X_data = f["X_data"][:]
        X_indices = f["X_indices"][:]
        X_indptr = f["X_indptr"][:]
        X_shape = tuple(f["X_shape"][:])

        # Restore sparse matrix
        X_loaded = scipy.sparse.csr_matrix((X_data, X_indices, X_indptr), shape=X_shape)

        # Load y
        y_loaded = f["y"][:]

    return X_loaded, y_loaded   

def load_dependence():
    with open(output_dependence_path.format("featureselector"), "rb") as f:
        selector = dill.load(f)
    with open(output_dependence_path.format("vectorizer"), "rb") as f:
        vectorizer = dill.load(f)

    return vectorizer, selector

    
def get_x_y(df, vectorizer=None, scaler=None, selector=None):
    df = na_imputing(df)

    # Separation der Zielvariable
    y = df['urgency']

    # separation von numerischen Attributen und Festsetzung auf float Vektor
    df_numeric = df.select_dtypes(include=['number', 'bool'])#.drop(columns= 'urgency')
    X_vec1 = np.array(df_numeric).astype(float)
    
    if scaler is None:
        scaler = MinMaxScaler()
        X_vec1 = scaler.fit_transform(X_vec1)
    else:
        X_vec1 = scaler.transform(X_vec1)



    if vectorizer is None:
        # Initialize TfidfVectorizer
        vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, ngram_range=(1, 3), tokenizer=tokenizer)
        # Fit the vectorizer to the data
        vectorizer.fit(df['combined_tks'])
        with open(output_dependence_path.format('vectorizer'), "wb") as f:
            dill.dump(vectorizer, f)  # Ensure the file is opened in 'wb' mode

    X_vec2 = vectorizer.transform(df['combined_tks'])

    if selector is None:
        k = 5000  # Number of best features to keep
        selector = SelectKBest(chi2, k=k)
        selector.fit(X_vec2, y)
        # safe selector to output dependence with dill
        with open(output_dependence_path.format('featureselector'), "wb") as f:
            dill.dump(selector, f)  # Ensure the file is opened in 'wb' mode

    X_vec2 = selector.transform(X_vec2)

    # Horizonatle Verbindung des Numerik-Vektors und Schlagwort-Vektors
    X_combined = hstack([X_vec1, X_vec2])


    
    return X_combined, y, vectorizer, scaler, selector

def get_x_y2(df, vectorizer=None, scaler=None, selector=None):
    df = na_imputing(df)

    # Separation der Zielvariable
    y = df['urgency']

    # separation von numerischen Attributen und Festsetzung auf float Vektor
    df_numeric = df.select_dtypes(include=['number', 'bool'])#.drop(columns= 'urgency')
    X_vec1 = np.array(df_numeric).astype(float)
    




    if vectorizer is None:
        # Initialize TfidfVectorizer
        vectorizer = TfidfVectorizer(max_df=1.0, min_df=1, ngram_range=(1, 3), tokenizer=tokenizer)
        # Fit the vectorizer to the data
        vectorizer.fit(df['combined_tks'])
        with open(output_dependence_path.format('vectorizer'), "wb") as f:
            dill.dump(vectorizer, f)  # Ensure the file is opened in 'wb' mode

    X_vec2 = vectorizer.transform(df['combined_tks'])

    if selector is None:
        k = 5000  # Number of best features to keep
        selector = SelectKBest(chi2, k=k)
        selector.fit(X_vec2, y)
        # safe selector to output dependence with dill
        with open(output_dependence_path.format('featureselector'), "wb") as f:
            dill.dump(selector, f)  # Ensure the file is opened in 'wb' mode

    X_vec2 = selector.transform(X_vec2)

    # Horizonatle Verbindung des Numerik-Vektors und Schlagwort-Vektors
    X_combined = hstack([X_vec1, X_vec2])

    if scaler is None:
        scaler = MinMaxScaler()
        X_vec1 = scaler.fit_transform(X_combined)
    else:
        X_combined = scaler.transform(X_combined)
    
    return X_combined, y, vectorizer, scaler, selector

def tokenizer(txt):
    return txt.split()

def filter_pre(df):
    mask = df.combined_tks.isna()
    df = df[~mask]
    mask = df.combined_tks.apply(str.split).apply(len) < 4
    df = df[~mask]
    return df

def filter_pre2(df):
    mask = df.combined_tks.isna()
    df = df[~mask]
    
    mask = (df.par_changes_count == 0)
    df = df[~mask]
    
    df.drop(columns=['par_changes_count'], inplace=True)
    
    return df


def na_imputing(df):
    df['combined_tks'].fillna(' ', inplace=True)
    return df


if __name__ == '__main__':
    get_sampled_data()