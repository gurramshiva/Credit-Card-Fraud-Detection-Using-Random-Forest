from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=50, max_depth=2, random_state=0, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)
