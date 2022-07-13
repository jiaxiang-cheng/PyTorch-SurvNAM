from run_nam import *


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

    test_scores = []
    while True:
        try:
            (x_train, y_train), (x_validate, y_validate) = next(train)
            model = train_model(x_train, y_train, x_validate, y_validate, device)
            metric, score = evaluate(model, test_loader, device)
            test_scores.append(score)
            logging.info(f"fold {len(test_scores)}/{n_splits} | test | {metric}={test_scores[-1]}")
        except StopIteration:
            break

        logging.info(f"mean test score={test_scores[-1]}")
