import os

import pandas as pd
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
    pbar = tqdm.tqdm(enumerate(data_loader, start=1), total=len(data_loader))
    total_loss = 0
    for i, (x, y) in pbar:
        # print("x", x[0][0].item())
        # print("x", x)

        def generate_normal(mean, std, N=100):
            s = np.random.normal(mean, std, N)
            return s

        [d_max, d_max_age, d_max_estrec, d_max_pnodes, d_max_progrec, d_max_tsize] = d_list

        gen_age = generate_normal(x[0][0].item(), d_max_age * 0.1)
        gen_estrec = generate_normal(x[0][1].item(), d_max_estrec * 0.1)
        gen_pnodes = generate_normal(x[0][4].item(), d_max_pnodes * 0.1)
        gen_progrec = generate_normal(x[0][5].item(), d_max_progrec * 0.1)
        gen_tsize = generate_normal(x[0][6].item(), d_max_tsize * 0.1)

        df_input = pd.concat([pd.DataFrame(gen_age),
                              pd.DataFrame(gen_estrec)], axis=1)

        df_input['horTh=yes'] = x[0][2].item()
        df_input['menostat=Post'] = x[0][3].item()

        df_input = pd.concat([df_input,
                              pd.DataFrame(gen_pnodes),
                              pd.DataFrame(gen_progrec),
                              pd.DataFrame(gen_tsize)], axis=1)

        df_input['tgrade'] = x[0][7].item()

        df_input.columns = ['age', 'estrec', 'horTh=yes', 'menostat=Post', 'pnodes', 'progrec', 'tsize', 'tgrade']
        # print(df_input)

        chf_rsf = rsf.predict_cumulative_hazard_function(df_input, return_array=True)

        for i, s in enumerate(chf_rsf):
            plt.step(rsf.event_times_, s, where="post", label=str(i))

        # print(chf_rsf[0])

        # plt.ylabel("Cumulative hazard")
        # plt.xlabel("Time in days")
        # # plt.legend()
        # plt.grid(True)
        # plt.show()

        input = df_input.iloc[0].astype(np.float32)
        input = torch.tensor(input.values)
        # print("input", input)
        x = torch.unsqueeze(input, dim=0)
        # print("x", x)
        x, y = x.to(device), y.to(device)
        # print("x", x)  # normalized input features of one instance

        logits, fnns_out = model.forward(x)
        # print("logits", logits)  # final additive outputs
        # print("fnns_out", fnns_out)  # outputs from each shape functions

        surv = rsf.predict_cumulative_hazard_function(x, return_array=True)

        loss = criterion(logits, surv, rsf.event_times_)

        total_loss -= (total_loss / i) - (loss.item() / i)

        model.zero_grad()

        loss.backward()

        optimizer.step()

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
    d_max = 0
    for i in range(X_train.shape[0]):
        for j in range(i):
            d_ij = sum((X_train.iloc[i] - X_train.iloc[0]) ** 2) ** 0.5
            d_max = d_ij if (d_ij >= d_max) else d_max

    d_max_age, d_max_estrec, d_max_pnodes, d_max_progrec, d_max_tsize = 0, 0, 0, 0, 0
    for i in range(X_train.shape[0]):
        for j in range(i):
            d_ij_age = abs(X_train.iloc[i].age - X_train.iloc[0].age)
            d_max_age = d_ij_age if (d_ij_age >= d_max_age) else d_max_age

            d_ij_estrec = abs(X_train.iloc[i].estrec - X_train.iloc[0].estrec)
            d_max_estrec = d_ij_estrec if (d_ij_estrec >= d_max_estrec) else d_max_estrec

            d_ij_pnodes = abs(X_train.iloc[i].pnodes - X_train.iloc[0].pnodes)
            d_max_pnodes = d_ij_age if (d_ij_pnodes >= d_max_pnodes) else d_max_pnodes

            d_ij_progrec = abs(X_train.iloc[i].progrec - X_train.iloc[0].progrec)
            d_max_progrec = d_ij_progrec if (d_ij_progrec >= d_max_progrec) else d_max_progrec

            d_ij_tsize = abs(X_train.iloc[i].tsize - X_train.iloc[0].tsize)
            d_max_tsize = d_ij_tsize if (d_ij_tsize >= d_max_tsize) else d_max_tsize

    print(d_max, d_max_age, d_max_estrec, d_max_pnodes, d_max_progrec, d_max_tsize)
    d_list = [d_max, d_max_age, d_max_estrec, d_max_pnodes, d_max_progrec, d_max_tsize]

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
            metric, score = evaluate(model, test_loader, device)
            test_scores.append(score)
            logging.info(f"fold {len(test_scores)}/{n_splits} | test | {metric}={test_scores[-1]}")
        except StopIteration:
            break

        logging.info(f"mean test score={test_scores[-1]}")
