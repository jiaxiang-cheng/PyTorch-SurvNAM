from base.rsf import *
from lifelines import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import time

from base.rsf.RandomSurvivalForest import RandomSurvivalForest
from base.rsf.scoring import concordance_index


def load_gbsg2():
    """

    :return:
    """
    gbsg2 = datasets.load_gbsg2()
    gbsg2_data = gbsg2[['age', 'tsize', 'pnodes', 'progrec', 'estrec']]

    gbsg2_horTh = pd.get_dummies(gbsg2.horTh, prefix='horTh')
    gbsg2_menostat = pd.get_dummies(gbsg2.menostat, prefix='menostat')
    gbsg2_tgrade = pd.get_dummies(gbsg2.tgrade, prefix='tgrade')
    gbsg2_data = pd.concat([gbsg2_data, gbsg2_horTh, gbsg2_menostat, gbsg2_tgrade], axis=1)
    gbsg2_target = gbsg2[['time', 'cens']]

    return gbsg2_data, gbsg2_target, "time", "cens"


def load_rossi():
    """

    :return:
    """
    rossi = datasets.load_rossi()
    # Attention: duration column must be index 0, event column index 1 in y
    return rossi.drop(["arrest", "week"], axis=1), rossi[["arrest", "week"]], "arrest", "week"


if __name__ == "__main__":

    X, y, duration, event = load_gbsg2()
    # X, y, duration, event = load_rossi()
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

    # print("RSF")
    start_time = time.time()
    rsf = RandomSurvivalForest(n_estimators=20, n_jobs=-1, min_leaf=10)
    # print("X", X, "y", y)
    rsf = rsf.fit(X, y)
    print("--- %s seconds ---" % (time.time() - start_time))
    y_pred = rsf.predict(X_test)
    # print(y_pred)
    c_val = concordance_index(y_time=y_test[duration], y_pred=y_pred, y_event=y_test[event])
    print("C-index", round(c_val, 3))
