
# Respecto a los puntos solicitados en el challengue latam

## Selección del modelo

1. Una vez ejecutado el notebbok exploration.ipynb,transcribi el modelo **XGBoost with Feature Importance and with Balance**,debido a que al analizar el *clasification_report()*,este modelo muestra las mejores puntuaciones *f1-score*,siendo especificamente de 0.66 y 0.37 para las clases 0 y 1,esto implica que es el mejor equilibrio entre las metricas de precision y recall de los modelos 
de XGB y Regresion logistica propuestos.Por lo cual este modelo tiene un accuracy menor(0.55) que otros modelos,pero clasifica de forma mas equilibrada la clase 0 y 1,a diferencia del **XGBoost with Feature Importance but without Balance** que tiene un accuracy de 0.81,pero tiene un recall(0.01) y f1-score(0.01) muy bajos.

En el contexto del challenge planteado,la predicción de ambas clases(0 y 1) es relevante,ya que se busca obtener e implementar un modelo que pueda pronosticar la variable "delay",por lo tanto es mejor seleccionar un modelo que permita balancear la preduccion de ambas clases.

## Transcripción del modelo

2. Luego al seleccionar el modelo **XGBoost with Feature Importance and with Balance**,lo lleve por secciones al archivo model.py,de esta forma en la primera funcion guarde el modelo,sin embargo como necesita del parametro *scale_pos_weight = scale*,transcribi en esta primera funcion parte del procesamiento de datos para generar el parametro *scale*.

3. Luego en las siguientes secciones separe el preprocesamiento,ajuste del modelo y predicciones,de acuerdo a las funciones:(1) preprocess,(2) fit y (3) predict.

4. Al final del documento *model.py*,realice algunas pruebas de la implementación del modelo.En donde el modelo si preprocesa los datos,ajusta y realiza predicciones incluso pudiendo verificar la matriz de confusion.

5. Sin embargo al momento de ejecutar el comando *make model-test*, solo se pasan 2 de las 4 pruebas.Sin embargo con un poco mas de tiempo,podria resolver este problema,debido a que al parecer cuando transcribo el modelo,el preprocesamiento no lo estoy haciendo de forma correcta.

6. Ya que el modelo no pasa las pruebas,no realice la implementación del modelo en fastapi.De igual forma,si tuviera mas tiempo para resolver las pruebas unitarias,podria desplegar el modelo con fastapi.

## Conclusiones

7. Finalmente,es un excelente desafio para postular a la vacante de software engineer(ML & LLm) y sin duda si tuviera la posibilidad de trabajar con el equipo junto a un poco mas de tiempo,podria abordar de mejor forma el desafio propuesto,ya que me encuentro motivado por trabajar en el desarrollo e implementación de modelos de Machine Learning,que generen valor tanto para los usuarios como para la empresa.

Saludos
