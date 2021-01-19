from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# loading data
X, y = load_iris(return_X_y=True)
X, y = X[:100], y[:100]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# linear
model_LS = RidgeClassifier(alpha=0)
model_LS.fit(X_train, y_train)
model_LS.predict(X_test)

train_score = model_LS.score(X_train, y_train)
test_score = model_LS.score(X_test, y_test)

print(f"(linear) model test accuracy: {test_score}, "\
      f"model train accuracy: {train_score}")
print(f"(linear) model coefficients: {model_LS.coef_}")

# logistic
model_LR = LogisticRegression(penalty='none', solver='lbfgs')
model_LR.fit(X_train, y_train)
model_LR.predict(X_test)

train_score_LR = model_LR.score(X_train, y_train)
test_score_LR = model_LR.score(X_test, y_test)

print(f"(logistic) model test accuracy: {test_score_LR}, "\
      f"model train accuracy: {train_score_LR}")
print(f"(logistic) model coefficients: {model_LS.coef_}")

# KNN
model_KNN = KNeighborsClassifier(n_neighbors=5)
model_KNN.fit(X_train, y_train)
model_KNN.predict(X_test)

train_score_KNN = model_KNN.score(X_train, y_train)
test_score_KNN = model_KNN.score(X_test, y_test)

print(f"(KNN) model test accuracy: {test_score_KNN}, "\
      f"model train accuracy: {train_score_KNN}")
print(f"(KNN) model coefficients: {model_KNN.classes_}")
