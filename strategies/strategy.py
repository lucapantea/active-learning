class Strategy:
    """
    A strategy class for active learning which handles querying, updating, fitting,
    and predicting using a given dataset and model.
    """

    def __init__(self, dataset, model) -> None:
        """
        Initializes the Strategy with a dataset and a model.
        """
        self.dataset = dataset
        self.model = model

    def query(self, n_query: int) -> None:
        """
        Queries the dataset based on the strategy.
        """
        pass 

    def update(self, pos_idxs: list, neg_idxs: list = None) -> None:
        """
        Updates the dataset with positive and optional negative indices.
        """
        self.dataset.update(pos_idxs, True)
        if neg_idxs:
            self.dataset.update(neg_idxs, False)

    def fit(self) -> None:
        """
        Fits the model using the labeled data from the dataset.
        """
        _, labeled_dataset = self.dataset.get_labeled_data()
        self.model.fit(labeled_dataset)

    def predict(self, test_dataset) -> list:
        """
        Predicts labels for the test dataset.
        """
        return self.model.predict(test_dataset)

    def predict_proba(self, test_dataset) -> list:
        """
        Predicts probabilities for the test dataset.
        """
        return self.model.predict_proba(test_dataset)

    def predict_prob_dropout(self, test_dataset, n_drop: int) -> list:
        """
        Predicts probabilities with dropout for the test dataset.
        """
        return self.model.predict_prob_dropout(test_dataset, n_drop)

    def predict_prob_dropout_split(self, data, n_drop: int) -> list:
        """
        Splits the data and predicts probabilities with dropout.
        """
        return self.model.predict_prob_dropout_split(data, n_drop)

    def get_embeddings(self, data) -> list:
        """
        Retrieves embeddings for the given data.
        """
        return self.model.get_embeddings(data)
