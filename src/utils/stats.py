#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 02:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   stats.py
# @Desc     :   

from json import load, dump
from numpy import ndarray, cumsum, argmax, random as np_random
from pandas import DataFrame, read_csv
from pprint import pprint
from pathlib import Path
from random import shuffle
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch import Tensor, tensor, float32

from src.utils.decorator import timer

WIDTH: int = 64


class NumpyRandomSeed:
    """ Setting random seed for reproducibility """

    def __init__(self, description: str, seed: int = 27):
        """ Initialise the RandomSeed class
        :param description: the description of a random seed
        :param seed: the seed value to be set
        """
        self._description: str = description
        self._seed: int = seed
        self._previous_np_seed = None

    def __enter__(self):
        """ Set the random seed """
        # Save the previous random seed state
        self._previous_np_seed = np_random.get_state()

        # Set the new random seed
        np_random.seed(self._seed)

        print("*" * 50)
        print(f"{self._description!r} has been set randomness {self._seed}.")
        print("-" * 50)

        return self

    def __exit__(self, *args):
        """ Exit the random seed context manager """
        # Restore the previous random seed state
        if self._previous_np_seed is not None:
            np_random.set_state(self._previous_np_seed)

        print("-" * 50)
        print(f"{self._description!r} has been restored to previous randomness.")
        print("*" * 50)
        print()

    def __repr__(self):
        """ Return a string representation of the random seed """
        return f"{self._description!r} is set to randomness {self._seed}."


@timer
def load_csv(csv_path: str) -> tuple[DataFrame, DataFrame]:
    """ Read data from a dataset file
    :param csv_path: path to the CSV file
    :return: data read from the file
    """
    dataset: DataFrame = read_csv(csv_path)

    y: DataFrame = dataset[:, -1]
    X: DataFrame = dataset.drop(dataset.columns[0], axis=1)

    print(f"X's type is {type(X)}, and its shape is {X.shape}.")
    print(f"y's type is {type(y)}, and its shape is {y.shape}.")

    return X, y


@timer
def load_text(text_data_path: str, cols: bool = False, columns: list | None = None) -> DataFrame:
    """ Read data from a txt file with a structural data format
    :param text_data_path: path to the text data file
    :param cols: whether to specify column names
    :param columns: list of column names
    :return: data read from the text file
    """
    if cols:
        data: DataFrame = read_csv(text_data_path, names=columns, sep=r"\s+")
    else:
        data: DataFrame = read_csv(text_data_path, sep=r"\s+")

    print(f"Loaded text data' shape is {data.shape}.")

    return data


@timer
def summary_dataframe(data: DataFrame) -> None:
    """ Print summary statistics of the data
    :param data: DataFrame containing the data
    """
    print(data.describe())
    print(f"Missing Values: {data.isnull().sum()[data.isnull().sum() > 0]}")
    print(f"Duplicated Rows: {data.duplicated().sum()}")


@timer
def load_paths(base_directory: Path) -> tuple[list[str], list[str]]:
    """ Load all file paths from a directory
    :param base_directory: path to the directory
    :return: list of file paths in the directory
    """
    images: list = []
    labels: list = []

    if base_directory.exists() and base_directory.is_dir():
        for subdir in base_directory.iterdir():
            match subdir.name:
                case "images":
                    for file in subdir.iterdir():
                        if file.is_file():
                            images.append(file.resolve())
                case "masks":
                    for file in subdir.iterdir():
                        if file.is_file():
                            labels.append(file.resolve())

        images.sort(key=lambda f: f.name)
        labels.sort(key=lambda f: f.name)

        print(f"Loaded {len(images)} files from each directory: {base_directory.name}")
    else:
        print(f"The directory {base_directory} does not exist or is not a directory.")

    return images, labels


@timer
def save_json(json_data: dict, json_path: str | Path) -> None:
    with open(str(json_path), "w", encoding="utf-8") as file:
        dump(json_data, file, indent=2)


@timer
def load_json(json_path: str | Path) -> dict:
    """ Load JSON data from a file
    :param json_path: path to the JSON file
    :return: data loaded from the JSON file
    """
    with open(str(json_path), "r", encoding="utf-8") as file:
        data: dict = load(file)

    print(f"JSON data loaded from {json_path}:")

    return data


@timer
def standardise_data(data: DataFrame, is_tensor: bool = False) -> tuple[DataFrame | Tensor, ColumnTransformer]:
    """ Preprocess the data by handling missing values, scaling numerical features, and encoding categorical features.
    :param data: the DataFrame containing the selected features for training
    :param is_tensor: whether to return a torch Tensor instead of a DataFrame
    :return: the preprocessed data and the fitted ColumnTransformer
    """
    # Divide the columns into numerical and categorical types
    cols_num: list[str] = data.select_dtypes(include=["int32", "int64", "float32", "float64"]).columns.tolist()
    cols_cat: list[str] = data.select_dtypes(include=["object", "category"]).columns.tolist()

    # Set a list of transformers to collect the pipelines
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    # Establish a pipe to process numerical features and handle missing values only if they exist
    if cols_num:
        pipe_num = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", pipe_num, cols_num))

    # Establish a pipe to process categorical features and handle missing values only if they exist
    if cols_cat:
        pipe_cat = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("cat", pipe_cat, cols_cat))

    # Establish a column transformer to process numerical and categorical features
    preprocessor: ColumnTransformer = ColumnTransformer(transformers=transformers)
    # Fit and transform the data
    out = preprocessor.fit_transform(data)

    # If the processed data is a sparse matrix, convert it to a dense array
    if hasattr(out, "toarray"):
        out: ndarray = out.toarray()

    # Return DataFrame or Tensor
    if not is_tensor:
        # Rebuild the DataFrame with processed data and proper column names
        output: DataFrame = DataFrame(data=out, columns=preprocessor.get_feature_names_out())
    else:
        # Build the torch tensor with processed data and proper column names
        # - tensor dtype is not quite suitable for PCA
        output: Tensor = tensor(out, dtype=float32)

    print(f"Preprocessed data type is {type(output)}, and its shape: {output.shape}")

    return output, preprocessor


@timer
def split_paths(image_paths: list, mask_paths: list, test_size: float = 0.2, shuffle_status: bool = True) -> tuple:
    """ Split the image and mask paths into training and validation sets
    :param image_paths: list of image file paths
    :param mask_paths: list of mask file paths
    :param test_size: the proportion of the dataset to include in the test split
    :param shuffle_status: whether to shuffle the data before splitting
    :return: the training and validation sets for image and mask paths
    """
    assert len(image_paths) == len(mask_paths), "The number of image paths must be equal to the number of mask paths."

    paired = list(zip(image_paths, mask_paths))
    if shuffle_status:
        shuffle(paired)
        # print(paired)

    split_index: int = int(len(image_paths) * (1 - test_size))
    paired_train_paths: list[tuple[Path, Path]] = paired[:split_index]
    paired_valid_paths: list[tuple[Path, Path]] = paired[split_index:]

    train_image_paths: list[Path] = [image for image, _ in paired_train_paths]
    train_mask_paths: list[Path] = [mask for _, mask in paired_train_paths]
    valid_image_paths: list[Path] = [image for image, _ in paired_valid_paths]
    valid_mask_paths: list[Path] = [mask for _, mask in paired_valid_paths]

    print(f"Training: {len(train_image_paths)} images, {len(train_mask_paths)} masks")
    print(f"Validation: {len(valid_image_paths)} images, {len(valid_mask_paths)} masks")

    return train_image_paths, train_mask_paths, valid_image_paths, valid_mask_paths


@timer
def split_array(
        features: DataFrame | list, labels: DataFrame | list,
        valid_size: float = 0.2, random_state: int = 27, shuffle_status: bool = True
) -> tuple:
    """ Split the data into training and testing sets
    :param features: the DataFrame containing the selected features for training
    :param labels: the DataFrame containing the target masks
    :param valid_size: the proportion of the dataset to include in the test split
    :param random_state: random seed for reproducibility
    :param shuffle_status: whether to shuffle the data before splitting
    :return: the training and testing sets for features and masks
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        features, labels,
        test_size=valid_size,
        random_state=random_state,
        shuffle=shuffle_status,
        stratify=None
    )

    if isinstance(X_train, DataFrame):
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}")
    else:
        print(f"X_train length: {len(X_train)}, y_train length: {len(y_train)}")
        print(f"X_valid length: {len(X_valid)}, y_valid length: {len(y_valid)}")

    return X_train, X_valid, y_train, y_valid


@timer
def select_pca_importance(data: DataFrame, threshold: float = 0.95, top_n: int = None) -> tuple[list, PCA, DataFrame]:
    """ Calculate PCA feature importance
    :param data: the DataFrame containing the selected features for training
    :param threshold: the cumulative variance ratio threshold to consider
    :param top_n: the number of top important features to return (if None, return all)
    :return: PCA feature importance scores
    """
    # Initialise PCA
    model = PCA()
    model.fit(data)

    # Calculate cumulative variance ratio
    cumulative_variance: ndarray = cumsum(model.explained_variance_ratio_)
    # Determine the number of components to reach the threshold
    n_components: int = int(argmax(cumulative_variance >= threshold) + 1)

    # Build a DataFrame to hold feature loadings
    ratios: DataFrame = DataFrame(
        model.components_.T,
        columns=[f"PC{i + 1}" for i in range(data.shape[1])],
        index=data.columns
    )

    # Calculate the absolute contribution of each feature to the selected components
    ratios["Contribution"] = ratios.iloc[:, :n_components].abs().sum(axis=1)
    # Sort features by their contribution
    ratios: DataFrame = ratios.sort_values("Contribution", ascending=False)
    # print(ratios)

    # Extract important features
    if top_n is not None:
        important_features = ratios.index[:top_n].tolist()
    else:
        important_features = ratios.index[:n_components].tolist()

    print(f"Features meet the threshold of {threshold * 100:.1f}% cumulative variance: {n_components}")
    print("Important Features:")
    pprint(important_features)
    print("Contribution of these features:")
    pprint(ratios.loc[important_features, "Contribution"].values)

    return important_features, model, ratios


if __name__ == "__main__":
    pass
