import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Union, List
import pickle
import os


class DelayModel:

    
    TOP_FEATURES = [
        "OPERA_Latin American Wings",
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    
    DELAY_THRESHOLD = 15

    def __init__(self):
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load pre-trained model"""
        model_path = "model/trained_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)

    @staticmethod
    def _get_min_diff(row: pd.Series) -> float:
        """
        Calculate the difference in minutes between scheduled and actual departure.

        Args:
            row: DataFrame row containing 'Fecha-O' and 'Fecha-I' columns.

        Returns:
            Difference in minutes between actual and scheduled departure.
        """
        fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        return ((fecha_o - fecha_i).total_seconds()) / 60

    def _generate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate one-hot encoded features from raw data.

        Args:
            data: Raw DataFrame with OPERA, TIPOVUELO, and MES columns.

        Returns:
            DataFrame with top 10 one-hot encoded features.
        """
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)

    
        for col in self.TOP_FEATURES:
            if col not in features.columns:
                features[col] = 0

        return features[self.TOP_FEATURES]

    def _generate_target(self, data: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Generate target column based on delay threshold.

        Args:
            data: Raw DataFrame with flight data.
            target_column: Name for the target column.

        Returns:
            DataFrame with binary target column.
        """
        if target_column not in data.columns:
            data["min_diff"] = data.apply(self._get_min_diff, axis=1)
            data[target_column] = np.where(
                data["min_diff"] > self.DELAY_THRESHOLD, 1, 0
            )
        return data[[target_column]]

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or prediction.

        Args:
            data: Raw DataFrame with flight data.
            target_column: If set, the target column is also returned.

        Returns:
            If target_column is set: Tuple of (features, target) DataFrames.
            Otherwise: Features DataFrame only.
        """
        features = self._generate_features(data)

        if target_column:
            target = self._generate_target(data, target_column)
            return features, target

        return features

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features: Preprocessed feature DataFrame.
            target: Target DataFrame.
        """
        target_series = target.iloc[:, 0]

       
        n_y0 = len(target_series[target_series == 0])
        n_y1 = len(target_series[target_series == 1])

        self._model = LogisticRegression(
            class_weight={1: n_y0 / len(target_series), 0: n_y1 / len(target_series)}
        )
        self._model.fit(features, target_series)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features: Preprocessed feature DataFrame.

        Returns:
            List of predicted targets (0 or 1).
        """
        predictions = self._model.predict(features)
        return predictions.tolist()
