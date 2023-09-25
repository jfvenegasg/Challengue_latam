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

        def get_period_day(date):
            date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
            morning_min = datetime.strptime("05:00", '%H:%M').time()
            morning_max = datetime.strptime("11:59", '%H:%M').time()
            afternoon_min = datetime.strptime("12:00", '%H:%M').time()
            afternoon_max = datetime.strptime("18:59", '%H:%M').time()
            evening_min = datetime.strptime("19:00", '%H:%M').time()
            evening_max = datetime.strptime("23:59", '%H:%M').time()
            night_min = datetime.strptime("00:00", '%H:%M').time()
            night_max = datetime.strptime("4:59", '%H:%M').time()
    
            if(date_time > morning_min and date_time < morning_max):
                return 'mañana'
            elif(date_time > afternoon_min and date_time < afternoon_max):
                return 'tarde'
            elif(
                (date_time > evening_min and date_time < evening_max) or
                (date_time > night_min and date_time < night_max)
            ):
                return 'noche'

        data['period_day'] = data['Fecha-I'].apply(get_period_day)

        def is_high_season(fecha):
            fecha_año = int(fecha.split('-')[0])
            fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
            range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
            range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
            range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
            range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
            range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
            range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
            range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
            range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
    
            if ((fecha >= range1_min and fecha <= range1_max) or 
                (fecha >= range2_min and fecha <= range2_max) or 
                (fecha >= range3_min and fecha <= range3_max) or
                (fecha >= range4_min and fecha <= range4_max)):
                return 1
            else:
                return 0

        data['high_season'] = data['Fecha-I'].apply(is_high_season)


        def get_min_diff(data):
            fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
            fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
            min_diff = ((fecha_o - fecha_i).total_seconds())/60
            return min_diff

        data['min_diff'] = data.apply(get_min_diff, axis = 1)
        threshold_in_minutes = 15  
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)     
        

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
        #data=pd.DataFrame(data)

        def get_period_day(date):
            date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
            morning_min = datetime.strptime("05:00", '%H:%M').time()
            morning_max = datetime.strptime("11:59", '%H:%M').time()
            afternoon_min = datetime.strptime("12:00", '%H:%M').time()
            afternoon_max = datetime.strptime("18:59", '%H:%M').time()
            evening_min = datetime.strptime("19:00", '%H:%M').time()
            evening_max = datetime.strptime("23:59", '%H:%M').time()
            night_min = datetime.strptime("00:00", '%H:%M').time()
            night_max = datetime.strptime("4:59", '%H:%M').time()
    
            if(date_time > morning_min and date_time < morning_max):
                return 'mañana'
            elif(date_time > afternoon_min and date_time < afternoon_max):
                return 'tarde'
            elif(
                (date_time > evening_min and date_time < evening_max) or
                (date_time > night_min and date_time < night_max)
            ):
                return 'noche'

        data['period_day'] = data['Fecha-I'].apply(get_period_day)

        def is_high_season(fecha):
            fecha_año = int(fecha.split('-')[0])
            fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
            range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
            range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
            range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
            range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
            range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
            range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
            range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
            range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
    
            if ((fecha >= range1_min and fecha <= range1_max) or 
                (fecha >= range2_min and fecha <= range2_max) or 
                (fecha >= range3_min and fecha <= range3_max) or
                (fecha >= range4_min and fecha <= range4_max)):
                return 1
            else:
                return 0

        data['high_season'] = data['Fecha-I'].apply(is_high_season)

        def get_min_diff(data):
            fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
            fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
            min_diff = ((fecha_o - fecha_i).total_seconds())/60
            return min_diff

        data['min_diff'] = data.apply(get_min_diff, axis = 1)
        threshold_in_minutes = 15  
        data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)     
        
        
        #data = shuffle(data[['OPERA', 'MES', 'TIPOVUELO', 'SIGLADES', 'DIANOM', 'delay']], random_state = 111)

        features = pd.concat([
        pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
        pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
        pd.get_dummies(data['MES'], prefix = 'MES')], 
        axis = 1
        )

        target = data['delay']


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
        #features=pd.DataFrame(features[top_10_features])
        features=features[top_10_features]
        
        features=pd.DataFrame(features)
        target=pd.DataFrame(target)

        #if target_column is not None:
            # Si 'target_column' no es None, separamos las características y el objetivo en dos DataFrames
        #features = pd.DataFrame(data.drop(columns=[target_column]))
        #target = pd.DataFrame(data[[target_column]])
        return features, target
        #else:
            # Si 'target_column' es None, devolvemos solo las características
        #    return data.drop(columns=[target_column])

    
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
        #data = pd.read_csv('data/data.csv', low_memory=False)
        #features,target = self.preprocess(data,target_column="delay")
        #features = self.preprocess(data=features)[0]
        #_, features_validation, _, target_validation = train_test_split(features, test_size = 0.33, random_state = 42)
        predictions = self._model.predict(features)
        return predictions.tolist()

# Cargar tus datos en un DataFrame de pandas
#data = pd.read_csv('data/data.csv', low_memory=False)

# Crear una instancia de la clase DelayModel
#model = DelayModel()

# Preprocesar los datos
#features_validation,target_validation = model.preprocess(data,target_column="delay")

# Entrenar el modelo
#model.fit(features_validation, target_validation)

# Realizar predicciones
#new_data = pd.read_csv("data/data.csv", low_memory=False)  # Reemplaza con tus nuevos datos
#predictions = model.predict(features_validation)
#print(predictions)

#from sklearn.metrics import confusion_matrix, classification_report

#print(confusion_matrix(target_validation, predictions))