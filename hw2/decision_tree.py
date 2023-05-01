import pandas as pd
import numpy as np
import math
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_data(train_path: str, test_path: str) -> tuple:
    """
    Reads in train and test data from specified file paths.

    :param train_path: File path of training data.
    :type train_path: str
    :param test_path: File path of test data.
    :type test_path: str
    :return: Tuple containing training and test data.
    :rtype: tuple
    """
    logger.info("Reading in data from file paths: %s, %s", train_path, test_path)

    # Read in data from file paths
    # wrap in try and catch block to handle file not found error
    try:
        train_data = pd.read_csv(train_path, delimiter="\t", header=None)
        test_data = pd.read_csv(test_path, delimiter="\t", header=None)
    except FileNotFoundError:
        logger.error("File not found.")
        sys.exit(1)

    # Set column names of test data to match those of train data
    test_data.columns = train_data.columns[:-1]

    return train_data, test_data


def entropy_calculation(data: pd.DataFrame) -> float:
    """
    Calculates entropy of target column in data.

    :param data: Dataframe containing target column.
    :type data: pd.DataFrame
    :return: Entropy of target column.
    :rtype: float
    """
    logger.info("Calculating entropy of target column.")

    # Get target column and calculate distribution of labels
    target_col = data.columns[-1]
    label_distribution = data[target_col].value_counts(normalize=True)

    return sum(-p * math.log2(p) for p in label_distribution)


def gain_ratio(data: pd.DataFrame, attribute: str) -> float:
    """
    Calculates gain ratio of specified attribute in data.

    :param data: Dataframe containing attribute and target columns.
    :type data: pd.DataFrame
    :param attribute: Name of attribute for which gain ratio is to be calculated.
    :type attribute: str
    :return: Gain ratio of specified attribute.
    :rtype: float
    """
    logger.info("Calculating gain ratio of attribute: %s", attribute)

    # Calculate initial entropy of data
    initial_entropy = entropy_calculation(data)

    # Group data by specified attribute
    grouped_data = data.groupby(attribute)

    # Initialize variables for calculating conditional entropy and split info
    conditional_entropy = 0
    split_info = 0

    # Iterate over groups and calculate conditional entropy and split info
    for _, subset in grouped_data:
        entropy_subset = entropy_calculation(subset)
        subset_weight = len(subset) / len(data)
        conditional_entropy += subset_weight * entropy_subset

        split_info += -subset_weight * math.log2(subset_weight)

    # Calculate information gain and gain ratio
    information_gain = initial_entropy - conditional_entropy

    return 0 if split_info == 0 else information_gain / split_info


def optimal_feature(data: pd.DataFrame) -> str:
    """
    Determines optimal feature for splitting data.

    :param data: Dataframe containing attributes and target column.
    :type data: pd.DataFrame
    :return: Name of optimal feature.
    :rtype: str
    """
    logger.info("Determining optimal feature for splitting data.")

    # Initialize variables for determining optimal feature
    top_feature = None
    highest_gain_ratio = -1

    # Iterate over attributes and calculate gain ratio
    for feature in data.columns[:-1]:
        current_gain_ratio = gain_ratio(data, feature)
        # Update top feature if current gain ratio is higher
        if current_gain_ratio > highest_gain_ratio:
            top_feature = feature
            highest_gain_ratio = current_gain_ratio
    return top_feature


def build_decision_tree(data: pd.DataFrame) -> dict:
    """
    Builds decision tree using data.

    :param data: Dataframe containing attributes and target column.
    :type data: pd.DataFrame
    :return: Decision tree.
    :rtype: dict
    """
    logger.info("Building decision tree.")

    # Get target column
    target_col = data.columns[-1]

    # If all labels are the same or data is empty, return mode of target column
    if len(data[target_col].unique()) == 1 or len(data) == 0:
        return data[target_col].mode().iloc[0]

    # Determine optimal feature for splitting data
    top_feature = optimal_feature(data)

    # Group data by optimal feature and recursively build decision tree
    grouped_data = data.groupby(top_feature)

    decision_tree = {top_feature: {}}

    for group_label, group in grouped_data:
        decision_tree[top_feature][group_label] = build_decision_tree(
            group.drop(top_feature, axis=1)
        )

    return decision_tree


def predict_single(sample: pd.Series, tree: dict) -> str:
    """
    Predicts label of single sample using decision tree.

    :param sample: Series containing attributes of sample.
    :type sample: pd.Series
    :param tree: Decision tree.
    :type tree: dict
    :return: Predicted label of sample.
    :rtype: str
    """
    root = list(tree.keys())[0]
    root_value = sample[root]
    subtree = tree[root].get(root_value)
    return predict_single(sample, subtree) if isinstance(subtree, dict) else subtree


def predict_all(test_data: pd.DataFrame, tree: dict) -> pd.DataFrame:
    """
    Predicts labels of all samples in test data using decision tree.

    :param test_data: Dataframe containing attributes of test data.
    :type test_data: pd.DataFrame
    :param tree: Decision tree.
    :type tree: dict
    :return: Dataframe containing predicted labels for each sample in test data.
    :rtype: pd.DataFrame
    """
    logger.info("Predicting labels for all samples in test data.")

    # Apply predict_single function to each row in test data and concatenate results
    predictions = test_data.apply(predict_single, axis=1, args=(tree,))
    return pd.concat([test_data, predictions], axis=1)


def output_results(output_path: str, result_data: pd.DataFrame) -> None:
    """
    Outputs results to specified file path.

    :param output_path: File path for output.
    :type output_path: str
    :param result_data: Dataframe containing results.
    :type result_data: pd.DataFrame
    """
    logger.info("Outputting results to file path: %s", output_path)

    # Write results to file
    result_data.to_csv(output_path, sep="\t", header=False, index=False)


def run_program(train_path: str, test_path: str, output_path: str) -> None:
    """
    Runs program to build decision tree and predict labels for test data.

    :param train_path: File path of training data.
    :type train_path: str
    :param test_path: File path of test data.
    :type test_path: str
    :param output_path: File path for output.
    :type output_path: str
    """
    logger.info("Running program.")

    # Read in data, build decision tree, predict labels, and output results
    train_data, test_data = read_data(train_path, test_path)
    decision_tree = build_decision_tree(train_data)
    result_data = predict_all(test_data, decision_tree)
    output_results(output_path, result_data)


if __name__ == "__main__":
    run_program(sys.argv[1], sys.argv[2], sys.argv[3])
