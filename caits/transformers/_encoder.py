from sklearn.base import BaseEstimator, TransformerMixin

from caits.dataset import Dataset


class LE(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.class_to_index = {}
        self.index_to_class = {}

    def fit(self, X: Dataset):
        """Fits the encoder to the dataset's target values.

        Args:
            dataset: A Dataset object containing the target values to encode.

        Returns:
            self: Returns the instance itself.
        """
        unique_classes = sorted(set(X.y))
        self.class_to_index = {label: index for index, label in enumerate(unique_classes)}
        self.index_to_class = {index: label for label, index in self.class_to_index.items()}
        return self

    def transform(self, X: Dataset) -> Dataset:
        """Transforms the dataset's target values to their encoded form.

        Args:
            dataset: A Dataset object containing the target values to encode.

        Returns:
            A new Dataset object with encoded target values.
        """
        encoded_labels = [self.class_to_index[label] for label in X.y]
        # Creating a new Dataset object with the encoded labels
        return Dataset(X.X, encoded_labels, X._id)

    def inverse_transform(self, dataset):
        """Transforms a dataset's encoded target values
        back to their original form.

        Args:
            dataset: A Dataset object containing encoded target values.

        Returns:
            A new Dataset object with the original target values.
        """
        original_labels = [self.index_to_class[index] for index in dataset.y]
        # Creating a new Dataset object with the original labels
        return Dataset(dataset.X, original_labels, dataset._id)
