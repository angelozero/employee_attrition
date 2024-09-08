# Employee Satisfaction Survey - Machine Learning

## Introdução

### Descrição do Problema e Objetivo do Projeto
O objetivo deste projeto é desenvolver um modelo de machine learning que prevê a satisfação dos funcionários com base em várias características coletadas em uma pesquisa. A previsão da satisfação pode ajudar a empresa a entender melhor as necessidades e expectativas dos seus colaboradores, promovendo um ambiente de trabalho mais saudável e produtivo.

### Breve Descrição do Dataset
O dataset "Employee Satisfaction Survey" contém informações relevantes sobre os funcionários, como níveis de satisfação, avaliações de desempenho, horas trabalhadas, tempo de serviço, acidentes de trabalho, promoções e salários. Com essas informações, podemos analisar os fatores que influenciam a satisfação dos funcionários.

### Dataset utilizado
[Kaggle - EmplolyeeSurvey](https://www.kaggle.com/code/dennismathewjose/emplolyeesurvey)

## Importação de Bibliotecas Necessárias
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
```

## Visualização Básica dos Dados
```python
# Carregando o dataset
df = pd.read_csv('/content/Employee Attrition.csv')

# Visualização básica dos dados
df.head(10)
```

### output
| Emp | ID | satisfaction_level  | last_evaluation	 | number_project | average_montly_hours | time_spend_company | work_accident | promotion_last_5years | dept	| salary |
| --- | -- | ------------------- | ----------------- | -------------- | -------------------- | ------------------ | ------------- | --------------------- | ------- | ------ |
| 0   |	1  |  0.38	             | 0.53	             | 2	          | 157	                 | 3	              | 0	          | 0	                  | sales	| low    |
| 1   |	2  |  0.80	             | 0.86	             | 5	          | 262	                 | 6	              | 0	          | 0	                  | sales	| medium |
| 2   |	3  |  0.11	             | 0.88	             | 7	          | 272	                 | 4	              | 0	          | 0	                  | sales	| medium |
| 3   |	4  |  0.72	             | 0.87	             | 5	          | 223	                 | 5	              | 0	          | 0	                  | sales	| low    |
| 4   |	5  |  0.37	             | 0.52	             | 2	          | 159	                 | 3	              | 0	          | 0	                  | sales	| low    |
| 5   |	6  |  0.41	             | 0.50	             | 2	          | 153	                 | 3	              | 0	          | 0	                  | sales	| low    |
| 6   |	7  |  0.10	             | 0.77	             | 6	          | 247	                 | 4	              | 0	          | 0	                  | sales	| low    |
| 7   |	8  |  0.92	             | 0.85	             | 5	          | 259	                 | 5	              | 0	          | 0	                  | sales	| low    |
| 8   |	9  |  0.89	             | 1.00	             | 5	          | 224	                 | 5	              | 0	          | 0	                  | sales	| low    |
| 9   |	10 |  0.42	             | 0.53	             | 2	          | 142	                 | 3	              | 0	          | 0	                  | sales	| low    |


## Pré-processamento dos Dados
### Limpeza de dados
```python
# Verificando valores ausentes
df.isnull().sum()

# Tratamento de valores ausentes (se necessário)
df.dropna(inplace=True)

# Remoção de duplicados
df.drop_duplicates(inplace=True)
```

## Separação entre variáveis independentes e dependentes
```python
# Supondo que a variável dependente seja 'Satisfaction'
X = df.drop('satisfaction_level', axis=1)
```

## Codificação de variáveis categóricas
```python
# Converte variáveis categóricas em variáveis dummy
X_encoded = pd.get_dummies(X, drop_first=True)

# Converte para categórica
y = (df['satisfaction_level'] >= 0.5).astype(int)
```

## Verificando a distribuição das classes antes do balanceamento
```python
print("Distribuição das classes antes do balanceamento:")
print(y.value_counts())
```

### output
```shell
Distribuição das classes antes do balanceamento:
satisfaction_level
0    709
1    290
Name: count, dtype: int64
```

## Balanceamento das classes
```python
y0 = df[df['satisfaction_level'] < 0.5]
y1 = df[df['satisfaction_level'] >= 0.5]
```

## Upsampling da classe minoritária
```python
y1_upsampled = resample(y1, replace=True, n_samples=len(y0), random_state=42)
data_balanced = pd.concat([y0, y1_upsampled])
```

## Atualizando as variáveis X e y
```python
y_balanced = (data_balanced['satisfaction_level'] >= 0.5).astype(int)
X_balanced = data_balanced.drop(['satisfaction_level'], axis=1)
X_encoded_balanced = pd.get_dummies(X_balanced, drop_first=True)
```

## Divisão dos dados em conjuntos de treino e teste (holdout)
```python
X_train, X_test, y_train, y_test = train_test_split(X_encoded_balanced, y_balanced, test_size=0.2, random_state=42)
```

## Transformação de Dados
### Normalização e padronização dos dados
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## Modelagem
### Criação de Pipelines para Cada Modelo
```python
# Criação de Pipelines para Cada Modelo
models = {
    'KNN': Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]),
    'Decision Tree': Pipeline([('scaler', StandardScaler()), ('tree', DecisionTreeClassifier())]),
    'Naive Bayes': Pipeline([('scaler', StandardScaler()), ('nb', GaussianNB())]),
    'SVM': Pipeline([('scaler', StandardScaler()), ('svm', SVC())])
}

results = {}
for model_name, model in models.items():
    model.fit(X_train_scaled, y_train)  # Treinamento do modelo
    y_pred = model.predict(X_test_scaled)  # Predição
    results[model_name] = accuracy_score(y_test, y_pred)  # Avaliação
```

## Otimização de Hiperparâmetros
### Uso de GridSearchCV
```python
# Definindo o grid de hiperparâmetros
param_grid = {
    'KNN': {'knn__n_neighbors': [3, 5, 7]},
    'Decision Tree': {'tree__max_depth': [None, 10, 20]},
    'Naive Bayes': {},  # Naive Bayes não tem hiperparâmetros relevantes para ajuste
    'SVM': {'svm__C': [0.1, 1, 10]}
}

best_models = {}
for model_name, model in models.items():
    # Verifica se há hiperparâmetros a serem ajustados
    if param_grid[model_name]:
        grid = GridSearchCV(model, param_grid[model_name], cv=5)
        grid.fit(X_train_scaled, y_train)
        best_models[model_name] = grid.best_estimator_
    else:
        # Treina o modelo sem ajuste de hiperparâmetros
        model.fit(X_train_scaled, y_train)
        best_models[model_name] = model
```

## Avaliação dos Modelos
### Comparação de Resultados
```python
for model_name, model in best_models.items():
    y_pred = model.predict(X_test_scaled)
    print(f"Model: {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n")
```

### output
```shell
Model: KNN
Accuracy: 0.9295774647887324
Classification Report:
               precision    recall  f1-score   support

           0       0.92      0.94      0.93       142
           1       0.94      0.92      0.93       142

    accuracy                           0.93       284
   macro avg       0.93      0.93      0.93       284
weighted avg       0.93      0.93      0.93       284

Confusion Matrix:
 [[134   8]
 [ 12 130]]


Model: Decision Tree
Accuracy: 0.9753521126760564
Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.96      0.98       142
           1       0.97      0.99      0.98       142

    accuracy                           0.98       284
   macro avg       0.98      0.98      0.98       284
weighted avg       0.98      0.98      0.98       284

Confusion Matrix:
 [[137   5]
 [  2 140]]


Model: Naive Bayes
Accuracy: 0.7676056338028169
Classification Report:
               precision    recall  f1-score   support

           0       0.81      0.70      0.75       142
           1       0.73      0.84      0.78       142

    accuracy                           0.77       284
   macro avg       0.77      0.77      0.77       284
weighted avg       0.77      0.77      0.77       284

Confusion Matrix:
 [[ 99  43]
 [ 23 119]]


Model: SVM
Accuracy: 0.9507042253521126
Classification Report:
               precision    recall  f1-score   support

           0       0.95      0.95      0.95       142
           1       0.95      0.95      0.95       142

    accuracy                           0.95       284
   macro avg       0.95      0.95      0.95       284
weighted avg       0.95      0.95      0.95       284

Confusion Matrix:
 [[135   7]
 [  7 135]]
```

## Exportação do Modelo
### Salvamento do Melhor Modelo
```python
feature_names = X_encoded.columns
joblib.dump(feature_names, 'best_model_employee_feature_names.pkl')

# Salvar o melhor modelo
joblib.dump(best_models['SVM'], 'best_model_employee_attrition.pkl')

# Para teste com PyTest
X_encoded_columns = X_encoded.columns.tolist()
joblib.dump((best_models['SVM'], X_encoded_columns), 'best_model_with_columns.pkl')
```

## Resumo dos Principais Achados

O projeto de previsão da satisfação dos funcionários revelou que o modelo de **Decision Tree** teve o melhor desempenho, com uma acurácia de **97,54%**. O **SVM** também apresentou bons resultados, com **95,07%** de acurácia. A distribuição dos dados mostrou que a maioria dos funcionários (709 de 999) estava insatisfeita, o que é preocupante. O upsampling da classe minoritária melhorou o balanceamento dos dados.

## Análise dos Resultados e Pontos de Atenção

Embora os resultados sejam positivos, o modelo Decision Tree pode sofrer de overfitting e requer monitoramento contínuo. Além disso, é essencial acompanhar as variáveis que afetam a satisfação dos funcionários e considerar a expansão do conjunto de dados para melhorar o modelo.

## Conclusão

Os modelos de machine learning demonstraram eficácia na previsão da satisfação dos funcionários, oferecendo insights valiosos para a empresa. A alta taxa de insatisfação identificada exige ações proativas para melhorar o ambiente de trabalho e aumentar a retenção e produtividade dos colaboradores. O monitoramento contínuo e a adaptação são cruciais para o sucesso a longo prazo.
