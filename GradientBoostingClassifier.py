import pandas as pd
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split


def gradient_boosting_classifier():

    X = pd.read_csv('df/bloomberg_mark_new_cluster_df.csv')[['cluster HDP']]
    print("X\n", X)
    y = pd.read_csv('df/bloomberg_mark_new_cluster_df.csv')['mark']
    print("y\n", y)
    X_train, X_test, y_train, y_true = train_test_split(X, y, test_size=0.2, random_state=1)
    model = GradientBoostingClassifier()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    print(f"\npredicted {y_predicted} \ntrue {y_true}")
    print(f"matches amount = {len([x for x, y in zip(y_predicted, y_true) if x == y])} from {len(y_predicted)}")
    print(metrics.confusion_matrix(y_true, y_predicted))
    print(metrics.classification_report(y_true, y_predicted, digits=3))


