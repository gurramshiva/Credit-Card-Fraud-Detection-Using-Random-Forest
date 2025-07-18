from sklearn.model_selection import train_test_split

def split_data(data):
    X = data.values[:, 0:29]
    y = data.values[:, 30]
    return train_test_split(X, y, test_size=0.3, random_state=0)
