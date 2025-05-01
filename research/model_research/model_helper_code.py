"""
Helper code to:
1. split datasets into training and testing dataset
2. cross validation
3. model evaluation
4. plot confusion matrix
"""

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
import matplotlib as plt
import seaborn as sns


def split_dataset(X, y, test_size=0.2):
    """
    Split the dataset into training and testing data.

    Args:
        X (DataFrame): dataset
        y (int): data labels
        test_size (float, optional): size of the testing dataset, defaults = 0.2

    Returns:
        DataFrame, DataFrame, Series, Series: training data, testing data, training data labels, testing data labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    print(f'Training datasets: {X_train.shape[0]} samples')
    print(f'Testing datasets: {X_test.shape[0]} samples')

    return X_train, X_test, y_train, y_test


def cross_val(model, X, y, cv=5):
    """
    Cross validations for the model.

    Args:
        model: machine learning model
        X (DataFrame): training data
        y (int): training data labels
        cv (int, optional): number of cross validation set, default = 5

    Returns:
        ndarray of float: scores of the estimator for each cross validation run
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    print(f'CV scores: {scores}')
    print(f'Mean CV accuracy: {scores.mean():.2f}')
    print(f'SD: {scores.std():.2f}')

    return scores


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.

    Args:
        model: Machine learning model
        X_test (DataFrame): Test datasets
        y_test (int): Test data labels

    Returns:
        tuple of float:
            loss: model loss
            accuracy: model accuracy
    """
    loss, accuracy = model.evaluate(X_test, y_test)

    print(f'Test Accuracy: {accuracy:.2f}')
    print(f'Test Loss: {loss:.2f}')

    return loss, accuracy


def plot_confusion_matrix(model, X_test, y_test):
    """
    Plot confusion matrix for the test data.

    Args:
        model: Machine learning model
        X_test (DataFrame): Test datasets
        y_test (int): Test data labels

    """
    # Predict labels for test data
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    plot = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plot.set_xlabel('Predicted labels')
    plot.set_ylabel('True or Physicians labels)')
    plot.set_title('Confusion Matrix')
    plt.show()
