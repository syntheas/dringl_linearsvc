from sklearn.svm import LinearSVC

def get_model():
    return LinearSVC(random_state=1)