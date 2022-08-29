import math
import os

import pandas as pd
import torch
import tqdm
import copy
import random
import logging
from absl import app
from absl import flags
from torch.utils.data import TensorDataset, DataLoader

# import base.nam.metrics
# from base.nam import data_utils
from base.nam import *

FLAGS = flags.FLAGS

learning_rate = 1e-3  # "Hyper-parameter: learning rate."
decay_rate = 0.995  # "Hyper-parameter: Optimizer decay rate"
output_regularization = 0.0  # "Hyper-parameter: feature reg"
l2_regularization = 0.0  # "Hyper-parameter: l2 weight decay"
dropout = 0.5  # "Hyper-parameter: Dropout rate"
feature_dropout = 0.0  # "Hyper-parameter: Prob. with which features are dropped"

training_epochs = 10  # "The number of epochs to run training for."
early_stopping_epochs = 60  # "Early stopping epochs"
batch_size = 1  # "Hyper-parameter: batch size."
data_split = 1  # "Dataset split index to use. Possible values are 1 to `num_splits`."
seed = 1  # "Seed used for reproducibility."
n_basis_functions = 1000  # "Number of basis functions to use in a FeatureNN for a real-valued feature."
units_multiplier = 2  # "Number of basis functions for a categorical feature"
n_models = 1  # "the number of models to train."
n_splits = 3  # "Number of data splits to use"
id_fold = 1  # "Index of the fold to be used"

hidden_units = []  # "Amounts of neurons for additional hidden layers, e.g. 64,32,32"
log_file = None  # "File where to store summaries."
dataset = "gbsg2"  # "Name of the dataset to load for training."
shallow_layer = "exu"  # "Activation function used for the first layer: (1) relu, (2) exu"
hidden_layer = "relu"  # "Activation function used for the hidden layers: (1) relu, (2) exu"
regression = True  # "Boolean for regression or classification"

n_folds = 5


def generate_normal(mean, std, N=100):
    """

    :param mean:
    :param std:
    :param N:
    :return:
    """
    s = np.random.normal(mean, std, N)
    return s


def seed_everything(seed):
    """

    :param seed:
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(x_train, y_train, x_valid, y_valid, device, rsf, d_list, nelson_est):
    """

    :param x_train:
    :param y_train:
    :param x_valid:
    :param y_valid:
    :param device:
    :return:
    """
    model = NeuralAdditiveModel(
        input_size=x_train.shape[-1],
        shallow_units=data_utils.calculate_n_units(x_train, n_basis_functions, units_multiplier),
        hidden_units=list(map(int, hidden_units)),
        shallow_layer=ExULayer if shallow_layer == "exu" else ReLULayer,
        hidden_layer=ExULayer if hidden_layer == "exu" else ReLULayer,
        hidden_dropout=dropout,
        feature_dropout=feature_dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=l2_regularization)
    # criterion = metrics.penalized_mse if regression else metrics.penalized_cross_entropy
    criterion = metrics.survnam_loss
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validate_dataset = TensorDataset(torch.tensor(x_valid), torch.tensor(y_valid))
    validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True)

    n_tries = early_stopping_epochs  # to restrict the minimum training epochs
    best_validation_score, best_weights = 0, None  # to store the optimal performance

    for epoch in range(training_epochs):
        model = model.train()  # training the base
        total_loss = train_one_epoch(model, criterion, optimizer, train_loader, device, rsf, d_list, nelson_est)
        # record the log of training (training loss)
        logging.info(f"epoch {epoch} | train | {total_loss}")

        scheduler.step()  # update the learning rate

        model = model.eval()  # validating the base
        metric, val_score, _ = evaluate(model, validate_loader, device)
        # record the log of validation (validation score)
        logging.info(f"epoch {epoch} | validate | {metric}={val_score}")

        # early stopping if the validation performance degrades
        # but also restricted to a minimum epochs of training
        if val_score <= best_validation_score and n_tries > 0:
            n_tries -= 1
            continue
        elif val_score <= best_validation_score:
            logging.info(f"early stopping at epoch {epoch}")
            break

        best_validation_score = val_score  # update the optimal validation score
        best_weights = copy.deepcopy(model.state_dict())  # update the optimal base

    model.load_state_dict(best_weights)  # continue training from the optimal base

    return model


def train_one_epoch(model, criterion, optimizer, data_loader, device, rsf, d_list, nelson_est):
    """

    :param model:
    :param criterion:
    :param optimizer:
    :param data_loader:
    :param device:
    :return:
    """

    # tqdm is a library in Python which is used for creating Progress Meters or Progress Bars.tqdm got its name from
    # the Arabic name taqaddum which means 'progress'.
    pbar = tqdm.tqdm(enumerate(data_loader, start=1), total=len(data_loader))

    total_loss = 0
    for i, (x, y) in pbar:
        x_loss = 0  # print("x", x[0][0].item()) # print("x", x)

        # ============================== Generate Points Following Normal Distribution =================================

        [_, d_max_age, d_max_estrec, d_max_pnodes, d_max_progrec, d_max_tsize] = d_list

        gen_age = generate_normal(x[0][0].item(), d_max_age * 0.1)
        gen_estrec = generate_normal(x[0][1].item(), d_max_estrec * 0.1)
        gen_pnodes = generate_normal(x[0][4].item(), d_max_pnodes * 0.1)
        gen_progrec = generate_normal(x[0][5].item(), d_max_progrec * 0.1)
        gen_tsize = generate_normal(x[0][6].item(), d_max_tsize * 0.1)

        df_input = pd.concat([pd.DataFrame(gen_age), pd.DataFrame(gen_estrec)], axis=1)
        df_input['horTh=yes'] = x[0][2].item()
        df_input['menostat=Post'] = x[0][3].item()
        df_input = pd.concat([df_input, pd.DataFrame(gen_pnodes), pd.DataFrame(gen_progrec),
                              pd.DataFrame(gen_tsize)], axis=1)
        df_input['tgrade'] = x[0][7].item()
        df_input.columns = ['age', 'estrec', 'horTh=yes', 'menostat=Post', 'pnodes', 'progrec', 'tsize', 'tgrade']
        # print(df_input)  generated N (=100) points for the explained point

        # largest distance among all generated points
        d_max = max_euclidean_distance(df_input)

        # input = df_input.iloc[0].astype(np.float32)
        # input = torch.tensor(input.values)
        # # print("input", input)
        # x = torch.unsqueeze(input, dim=0)
        # # print("x", x)
        # x, y = x.to(device), y.to(device)
        # # print("x", x)  # normalized input features of one instance

        # logits, fnns_out = model.forward(x)
        # print("logits", logits)  # final additive outputs
        # print("fnns_out", fnns_out)  # outputs from each shape functions
        # print("logits", logits)  # logits tensor([1.1540], grad_fn=<AddBackward0>)

        # print("nelson_est", nelson_est, len(nelson_est), type(nelson_est))  # length = 454
        # 18.0      0.000000
        #             ...
        # 2471.0    1.028009
        # 2551.0    1.028009

        # print("rsf.event_times_", rsf.event_times_, len(rsf.event_times_), type(rsf.event_times_))  # length = 215
        # [72.   98.  120.  160.  169.  171.  173.  177.  180.  184.  191.  195.
        #  205.  223.  227.  233.  238.  241.  242.  247.  249.  272.  281.  286.
        #  288.  293.  307.  308.  329.  336.  338.  344.  348.  350.  357.  359.
        #  360.  369.  370.  372.  374.  375.  385.  392.  394.  403.  410.  415. ...

        # print("chf_rsf[0]", chf_rsf[0], len(chf_rsf[0]), type(chf_rsf[0]))  # length = 215
        # [0.00442582 0.00446031 0.00449479 0.00630778 0.00834145 0.00841288
        #  0.00850506 0.00923685 0.01042667 0.01404961 0.02457283 0.02506429
        #  0.02506429 0.0356179  0.03565238 0.03629387 0.03876501 0.03909379 ...

        # CHF estimated by RSF for generated points (N = 100)
        chf_rsf = rsf.predict_cumulative_hazard_function(df_input, return_array=True)
        # print(chf_rsf[0])  CHF of 0th instance by RSF
        # [0.00207913 0.00207913 0.00229458 0.00318554 0.00665146 0.00665146
        #  0.01027261 0.01223869 0.01479719 0.01945229 0.02890883 0.0289664
        #  0.03129134 0.03592671 0.037428   0.0419167  0.04567486 0.04567486 ...

        # find all estimated CHF by nelson-aalon estimator corresponding to time points with events
        list_nelson = []
        for _, event_time in enumerate(rsf.event_times_):
            # print(nelson_est.loc[72])  # 0.001988071570576899
            list_nelson.append(nelson_est.loc[event_time])
        # append the CHF by nelson-aalon estimator
        est_chf = pd.concat([pd.DataFrame(rsf.event_times_), pd.DataFrame(list_nelson)], axis=1)

        # append all CHF by random survival forest for generated points
        for _, chf_generated_point in enumerate(chf_rsf):
            est_chf = pd.concat([est_chf, pd.DataFrame(chf_generated_point)], axis=1)
            # plt.step(rsf.event_times_, s, where="post", label=str(i))

        # formulate the column names
        column_list = ['event_times', 'chf_nelson']
        for num in range(100):
            column_list.append('chf_rsf_{}'.format(num))
        est_chf.columns = column_list

        # print(est_chf)
        #     event_times chf_rsf     chf_nelson
        # 0   72.0        0.001709    0.001988
        # 1   98.0        0.001709    0.003980
        # 2   120.0       0.002140    0.005980
        # 3   160.0       0.002745    0.007984
        # 4   169.0       0.007239    0.009996

        # surv = rsf.predict_cumulative_hazard_function(x, return_array=True)
        # loss = 0

        # df_temp = df_input.copy()
        df_input.loc[len(df_input)] = x[0].tolist()  # append the explained point to the end
        # print("df_temp", df_temp)
        # print("pd.Series(x[0])", pd.Series(x[0]))

        for k in range(100):

            xk = df_input.iloc[k].astype(np.float32)
            xk = torch.unsqueeze(torch.tensor(xk.values), dim=0)
            # x, y = x.to(device), y.to(device)
            logits, fnns_out = model.forward(xk)

            dx = sum((df_input.iloc[k] - df_input.iloc[-1]) ** 2) ** 0.5
            weight_i = 1 - (dx / d_max) ** 0.5
            # print("weight_i", weight_i)

            for j in range(215):
                if j == 0:
                    duration_j = est_chf['event_times'].loc[j] / est_chf['event_times'].loc[214]
                    phi = 0
                else:
                    duration_j = (est_chf['event_times'].loc[j] - est_chf['event_times'].loc[j - 1]) \
                                 / est_chf['event_times'].loc[214]
                    phi = math.log(est_chf['chf_rsf_{}'.format(k)].loc[j]) - math.log(est_chf['chf_nelson'].loc[j])

                logits_j = logits * ((weight_i * duration_j) ** 0.5)
                truths = phi * ((weight_i * duration_j) ** 0.5)
                truths = torch.as_tensor(truths)

                loss = criterion(logits_j, truths)
                loss.backward(retain_graph=True)
                x_loss += loss.item()

        # print("x_loss:", x_loss)
        # loss = criterion(logits, est_chf)

        # total_loss -= (total_loss / i) - (loss.item() / i)
        total_loss += x_loss

        optimizer.step()
        model.zero_grad()

        pbar.set_description(f"train | loss = {total_loss:.5f}")

    return total_loss


def evaluate(model, data_loader, device):
    """

    :param model:
    :param data_loader:
    :param device:
    :return:
    """
    total_score = 0
    metric = None
    for i, (x, y) in enumerate(data_loader, start=1):
        x, y = x.to(device), y.to(device)
        logits, fnns_out = model.forward(x)
        metric, score = metrics.calculate_metric(logits, y, regression=regression)
        total_score -= (total_score / i) - (score / i)

    return metric, total_score, logits


def max_euclidean_distance(df):
    """
    Find the maximum Euclidean distance among instances in dataframe
    :param df: input dataframe with all instance
    :return: max distance among samples
    """
    max_distance = 0
    for former in range(df.shape[0]):
        for latter in range(former + 1, df.shape[0]):
            temp_distance = sum((df.iloc[latter] - df.iloc[former]) ** 2) ** 0.5
            max_distance = temp_distance if (temp_distance >= max_distance) else max_distance
    return max_distance


def max_variable_distances(df, variable_list):
    """
    Find the maximum absolute distances among selected variables
    :param df: input dataframe with all instances
    :param variable_list: selected variables for finding distances
    :return: max distances of variables
    """
    max_distance = {}
    for _, variable in enumerate(numeric_list):
        max_distance[variable] = 0

    for former in range(df.shape[0]):
        for latter in range(former + 1, df.shape[0]):
            temp_distance = abs(df.iloc[latter] - df.iloc[former])
            for _, variable in enumerate(variable_list):
                max_distance[variable] = temp_distance[variable] if (temp_distance[variable] > max_distance[variable]) \
                    else max_distance[variable]
    return max_distance


if __name__ == "__main__":
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

    # ========================= Train Random Survival Forests ==========================================================
    X, y = load_gbsg2()

    grade_str = X.loc[:, "tgrade"].astype(object).values[:, np.newaxis]
    grade_num = OrdinalEncoder(categories=[["I", "II", "III"]]).fit_transform(grade_str)

    X_no_grade = X.drop("tgrade", axis=1)
    Xt = OneHotEncoder().fit_transform(X_no_grade)
    Xt.loc[:, "tgrade"] = grade_num

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    Xt = pd.DataFrame(scaler.fit_transform(Xt), columns=Xt.columns)

    random_state = 20
    X_train, X_test, y_train, y_test = train_test_split(Xt, y, test_size=0.25, random_state=random_state)
    print(X_train.head(3))

    # ============ Find Max Distances ========
    d_max = max_euclidean_distance(X_train)  # max euclidean distance among all samples after normalization

    numeric_list = ['age', 'estrec', 'pnodes', 'progrec', 'tsize']
    max_distances = max_variable_distances(X_train, numeric_list)
    d_list = [d_max, max_distances['age'], max_distances['estrec'],
              max_distances['pnodes'], max_distances['progrec'], max_distances['tsize']]
    # print(d_list)  # [2.2164908688956717, 1.0, 0.9265734265734266, 1.0, 1.0, 1.0000000000000002]

    rsf = RandomSurvivalForest(n_estimators=1000,
                               min_samples_split=10,
                               min_samples_leaf=15,
                               max_features="sqrt",
                               n_jobs=-1,
                               random_state=random_state)
    rsf.fit(X_train, y_train)

    X_test_sorted = X_test.sort_values(by=["pnodes", "age"])
    X_test_sel = pd.concat((X_test_sorted.head(3), X_test_sorted.tail(3)))
    # pd.Series(rsf.predict(X_test_sel))

    surv = rsf.predict_cumulative_hazard_function(X_test_sel, return_array=True)

    for i, s in enumerate(surv):
        plt.step(rsf.event_times_, s, where="post", label=str(i))

    plt.ylabel("Cumulative hazard")
    plt.xlabel("Time in days")
    plt.legend()
    plt.grid(True)
    plt.show()

    # =========================== Nelson-Aalon Estimator ===============================================================
    from lifelines.fitters.nelson_aalen_fitter import NelsonAalenFitter

    nelson = NelsonAalenFitter()
    nelson.fit(durations=pd.DataFrame(y_train).time, event_observed=pd.DataFrame(y_train).cens)

    x_nelson = nelson.cumulative_hazard_.index
    y_nelson = nelson.cumulative_hazard_.NA_estimate

    plt.step(nelson.cumulative_hazard_.index,
             nelson.cumulative_hazard_.NA_estimate,
             where="post",
             label=str(i))

    plt.ylabel("Cumulative hazard")
    plt.xlabel("Time in days")
    # plt.legend()
    plt.grid(True)
    plt.show()

    # =========================== Train Neural Additive Model ==========================================================
    test_scores = []
    while True:
        try:
            (x_train, y_train), (x_validate, y_validate) = next(train)
            model = train_model(x_train, y_train, x_validate, y_validate, device, rsf, d_list, y_nelson)
            # metric, score = evaluate(model, test_loader, device)
            # test_scores.append(score)
            # logging.info(f"fold {len(test_scores)}/{n_splits} | test | {metric}={test_scores[-1]}")
        except StopIteration:
            break

        # logging.info(f"mean test score={test_scores[-1]}")
