import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Union, List
import pickle 
import os

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.

        model_path = "data/trained_model.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)


    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix = 'MES')],
            axis = 1
        )

        #Top 10 columns
        top_10_features = [
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

        # Asegurar que existan todas las columnas necesarias
        for col in top_10_features:
            if col not in features.columns:
                features[col] = 0

        features = features[top_10_features]
        
        def get_min_diff(data):
            fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
            fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
            min_diff = ((fecha_o - fecha_i).total_seconds())/60
            return min_diff
        

        if target_column:
            if target_column not in data.columns:
                data["min_diff"] = data.apply(get_min_diff, axis = 1)
                threshold_in_minutes = 15
                data[target_column] = np.where(data["min_diff"] > threshold_in_minutes, 1, 0)
        
            target = data[[target_column]]
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
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        # Convertir target DataFrame a Series
        target_series = target.iloc[:, 0]

        # Calcular class weights para balancear clases
        n_y0 = len(target_series[target_series == 0])
        n_y1 = len(target_series[target_series == 1])

        self._model = LogisticRegression(class_weight={1: n_y0/len(target_series), 0: n_y1/len(target_series)})

        # Entrenar con Series en lugar de DataFrame
        self._model.fit(features, target_series)

        return 

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """

        predictions = self._model.predict(features)

        return predictions.tolist()