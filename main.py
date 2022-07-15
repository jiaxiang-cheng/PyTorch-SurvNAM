from run_nam import *
from demo_rsf import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest


def train_rsf():
    """

    :return:
    """
    # collecting data for training and testing
    X, y, duration, event = load_gbsg2()
    X, X_test, y, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

    # start training
    start_time = time.time()
    rsf_model = RandomSurvivalForest(n_estimators=20, n_jobs=-1, min_leaf=10)
    rsf_model = rsf_model.fit(X, y)
    print("--- %s seconds ---" % (time.time() - start_time))

    # start testing
    y_pred = rsf_model.predict(X_test)
    c_val = concordance_index(y_time=y_test[duration], y_pred=y_pred, y_event=y_test[event])
    print("C-index", round(c_val, 3))

    return rsf_model


if __name__ == "__main__":

    # First, we need to load the data and transform it into numeric values.
    X, y = load_gbsg2()

    grade_str = X.loc[:, "tgrade"].astype(object).values[:, np.newaxis]
    grade_num = OrdinalEncoder(categories=[["I", "II", "III"]]).fit_transform(grade_str)

    X_no_grade = X.drop("tgrade", axis=1)
    Xt = OneHotEncoder().fit_transform(X_no_grade)
    Xt.loc[:, "tgrade"] = grade_num

    # Next, the data is split into 75% for training and 25% for testing,
    # so we can determine how well our model generalizes.
    random_state = 20
    X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.25, random_state=random_state)

    # Training =========================================================================================================
    rsf = RandomSurvivalForest(n_estimators=1000,
                               min_samples_split=10,
                               min_samples_leaf=15,
                               max_features="sqrt",
                               n_jobs=-1,
                               random_state=random_state)
    rsf.fit(X_train, y_train)

    # We can check how well the model performs by evaluating it on the test data.
    rsf.score(X_test, y_test)

    # Predicting =======================================================================================================
    # X_test_sorted = X_test.sort_values(by=["pnodes", "age"])
    # X_test_sel = X_test_sorted.head(1)
    # pd.Series(rsf.predict(X_test_sel))
    # surv = rsf.predict_cumulative_hazard_function(X_test_sel, return_array=True)

    # ==================================================================================================================
    seed_everything(seed)  # random seed

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers)

    # cpu or gpu to train the base
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("load data")

    train, (x_test, y_test) = data_utils.create_test_train_fold(dataset=dataset,
                                                                id_fold=id_fold,
                                                                n_folds=n_folds,
                                                                n_splits=n_splits,
                                                                regression=not regression)
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    logging.info("begin training")

    test_scores = []
    while True:
        try:
            (x_train, y_train), (x_validate, y_validate) = next(train)
            model = train_model(x_train, y_train, x_validate, y_validate, device, rsf)
            metric, score, logits = evaluate(model, test_loader, device)
            test_scores.append(score)
            logging.info(f"fold {len(test_scores)}/{n_splits} | test | {metric}={test_scores[-1]}")
        except StopIteration:
            break

        logging.info(f"mean test score={test_scores[-1]}")
