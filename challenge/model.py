import numpy as np
import pandas as pd
from datetime import datetime

from typing import Tuple, Union, List

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

class DelayModel:

    def __init__(
        self
    ):  
        data=pd.read_csv('data/data.csv', low_memory=False)
        def get_min_diff(data):
            fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
            fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
            min_diff = ((fecha_o - fecha_i).total_seconds())/60
            return min_diff

        data['min_diff'] = data.apply(get_min_diff, axis = 1)
        threshold_in_minutes = 15  
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)     
        
        
        training_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state = 111)

        features = pd.concat([
        pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
        pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
        pd.get_dummies(data['MES'], prefix = 'MES')], 
        axis = 1
        )

        target = data['delay']

        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        scale = n_y0/n_y1

        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = scale) # Model should be saved in this attribute.

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
      
        def get_min_diff(data):
            fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
            fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
            min_diff = ((fecha_o - fecha_i).total_seconds())/60
            return min_diff

        data['min_diff'] = data.apply(get_min_diff, axis = 1)
        threshold_in_minutes = 15  
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)     
        
        
        training_data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state = 111)

        features = pd.concat([
        pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
        pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
        pd.get_dummies(data['MES'], prefix = 'MES')], 
        axis = 1
        )

        target = data['delay']

        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        scale = n_y0/n_y1

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
        features=features[top_10_features]
        target=target
        #features_validation,target_validation = train_test_split(features[top_10_features],target, test_size = 0.33, random_state = 42)

        #x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.33, random_state = 42)

        
        #return x_train, x_test, y_train, y_test
        return features,target
    
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
        self._model.fit(features,target)
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
        return predictions

# Cargar tus datos en un DataFrame de pandas
data = pd.read_csv('data/data.csv', low_memory=False)

# Crear una instancia de la clase DelayModel
model = DelayModel()

# Preprocesar los datos
features_validation,target_validation = model.preprocess(data,target_column="delay")

# Entrenar el modelo
model.fit(features_validation, target_validation)

# Realizar predicciones
new_data = pd.read_csv("data/data.csv", low_memory=False)  # Reemplaza con tus nuevos datos
predictions = model.predict(features_validation)
#print(predictions)

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(target_validation, predictions))