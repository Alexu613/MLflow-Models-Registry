# Import necessary libraries :
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

import warnings
warnings.filterwarnings("ignore")

class AutomaticMLflowLogger:
    """
    AutomaticMLflowLogger class:
    Generates a synthetic classification dataset, converts it to DataFrame, splits it into train/test sets,
    and logs a LogisticRegression model with MLflow.
    """

    def __init__(self, n_samples:int, n_features:int, n_informative:int, n_redundant:int, random_state:int, n_classes:int) -> None:
        """
        Initializes an instance of AutomaticMLflowLogger.

        Parameters:
        -----------
        n_samples : int
            Total number of samples to generate (e.g., 1000).
        n_features : int
            Total number of features (e.g., 20).
        n_informative : int
            Number of informative features (e.g., 10).
        n_redundant : int
            Number of redundant features (e.g., 5).
        random_state : int
            Random seed for reproducibility (e.g., 42).
        n_classes : int
            Number of classes in the target variable (e.g., 3).
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_informative = n_informative
        self.n_redundant = n_redundant
        self.random_state = random_state
        self.n_classes = n_classes

        # Generate classification dataset
        self.X, self.y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            random_state=random_state,
            n_classes=n_classes
        )

        # Print dataset info
        print("INFORMATION ON THE GENERATED DATASET:")
        print(f"--------------------------------------")
        print(f"X dtype: {self.X.dtype}, y dtype: {self.y.dtype}")
        print(f"Number of samples: {n_samples}, Features: {n_features}, Classes: {n_classes}")
        print(f"First 5 rows of X:\n{self.X[:5]}")
        print(f"First 5 rows of y:\n{self.y[:5]}")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert X and y into a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature columns named feature_0, ..., feature_n and target column named 'target'.
        """
        self.df = pd.DataFrame(data=self.X, columns=[f"feature_{n}" for n in range(self.X.shape[1])])
        self.df["target"] = self.y
        return self.df

    def split_dataframe(self, test_size:float=0.2, random_state:int=None) -> None:
        """
        Split the dataset into train and test sets using sklearn's train_test_split.

        Parameters
        ----------
        test_size : float
            Proportion of the dataset to include in the test split (default=0.2)
        random_state : int
            Random seed for reproducibility
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
        print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")

    def tracking_params_with_mlflow(self, name_experiment: str, penalty="l2", solver="lbfgs", random_state=42, n_jobs=1, run_name="run"):
        """
        Train a LogisticRegression model and log it with MLflow.

        Parameters:
        -----------
        name_experiment : str
            Name of the MLflow experiment.
        penalty : str
            Regularization type (default: 'l2').
        solver : str
            Solver for LogisticRegression (default: 'lbfgs').
        random_state : int
            Random state for reproducibility (default: 42).
        n_jobs : int
            Number of parallel jobs (default: 1).
        run_name : str
            Name of the MLflow run (default: 'run').
        """
        try:
            # Set experiment
            mlflow.set_experiment(name_experiment)
            mlflow.sklearn.autolog()  # Enable automatic logging

            # Start run
            with mlflow.start_run(run_name=run_name):
                model = LogisticRegression(penalty=penalty, solver=solver, random_state=random_state, n_jobs=n_jobs)
                model.fit(self.X_train, self.y_train)

                # Predictions
                y_train_pred = model.predict(self.X_train)
                y_test_pred = model.predict(self.X_test)

                # Metrics
                train_acc = accuracy_score(self.y_train, y_train_pred)
                test_acc = accuracy_score(self.y_test, y_test_pred)
                print(f"Train Accuracy: {train_acc:.4f}")
                print(f"Test Accuracy: {test_acc:.4f}")

                mlflow.sklearn.log_model(model, "model")
                run = mlflow.last_active_run()
                print(run.info.run_id)

        except Exception as e:
            print(f"Error in MLflow logging: {e}")