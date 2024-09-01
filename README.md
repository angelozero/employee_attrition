# Employee Satisfaction Survey - Machine Learning

## Introdução

### Descrição do Problema e Objetivo do Projeto
O objetivo deste projeto é desenvolver um modelo de machine learning que prevê a satisfação dos funcionários com base em várias características coletadas em uma pesquisa. A previsão da satisfação pode ajudar a empresa a entender melhor as necessidades e expectativas dos seus colaboradores, promovendo um ambiente de trabalho mais saudável e produtivo.

### Breve Descrição do Dataset
O dataset "Employee Satisfaction Survey" contém informações relevantes sobre os funcionários, como níveis de satisfação, avaliações de desempenho, horas trabalhadas, tempo de serviço, acidentes de trabalho, promoções e salários. Com essas informações, podemos analisar os fatores que influenciam a satisfação dos funcionários.

## Carga de Dados

### Importação de Bibliotecas Necessárias
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

# Carregando o dataset
```python
url = 'https://github.com/angelozero/machine_learning/blob/main/Employee%20Attrition.csv' 
data = pd.read_csv(url)
```

# Visualização Básica dos Dados
```python
print(data.head())
print(data.info())
```

# Pré-processamento dos Dados
```python
# Tratamento de valores ausentes
data.dropna(inplace=True)

# Remover duplicatas
data.drop_duplicates(inplace=True)
```

# Separação entre Variáveis Independentes e Dependentes
```python
X = data.drop(['satisfaction_level'], axis=1)

# Codificação de variáveis categóricas
X_encoded = pd.get_dummies(X, drop_first=True)  

# Variável alvo categórica
y = (data['satisfaction_level'] >= 0.5).astype(int)
```

# Divisão dos Dados em Conjuntos de Treino e Teste (Holdout)
```python
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
```

# Transformação de Dados
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

# Modelagem
### Criação de Pipelines para Cada Modelo
```python
models = {
    'KNN': Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]),
    'Decision Tree': Pipeline([('scaler', StandardScaler()), ('tree', DecisionTreeClassifier())]),
    'Naive Bayes': Pipeline([('scaler', StandardScaler()), ('nb', GaussianNB())]),
    'SVM': Pipeline([('scaler', StandardScaler()), ('svm', SVC())])
}
```

# Otimização de Hiperparâmetros
```python
param_grid = {
    'KNN': {'knn__n_neighbors': [3, 5, 7]},
    'Decision Tree': {'tree__max_depth': [None, 10, 20]},
    'Naive Bayes': {}, 
    'SVM': {'svm__C': [0.1, 1, 10]}
}

best_models = {}
for model_name, model in models.items():
    if param_grid[model_name]:
        grid = GridSearchCV(model, param_grid[model_name], cv=5)
        grid.fit(X_train_scaled, y_train)
        best_models[model_name] = grid.best_estimator_
    else:
        model.fit(X_train_scaled, y_train)
        best_models[model_name] = model
```

# Avaliação dos Modelos
```python
for model_name, model in best_models.items():
    y_pred = model.predict(X_test_scaled)
    print(f"Model: {model_name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\n")
```
### *Output*
```shell
Model: KNN
Accuracy: 0.8376666666666667
Classification Report:
               precision    recall  f1-score   support

           0       0.83      0.59      0.69       911
           1       0.84      0.95      0.89      2089

    accuracy                           0.84      3000
   macro avg       0.83      0.77      0.79      3000
weighted avg       0.84      0.84      0.83      3000

Confusion Matrix:
 [[ 534  377]
 [ 110 1979]]


Model: Decision Tree
Accuracy: 0.8503333333333334
Classification Report:
               precision    recall  f1-score   support

           0       0.80      0.67      0.73       911
           1       0.87      0.93      0.90      2089

    accuracy                           0.85      3000
   macro avg       0.83      0.80      0.81      3000
weighted avg       0.85      0.85      0.85      3000

Confusion Matrix:
 [[ 614  297]
 [ 152 1937]]


Model: Naive Bayes
Accuracy: 0.7963333333333333
Classification Report:
               precision    recall  f1-score   support

           0       0.67      0.65      0.66       911
           1       0.85      0.86      0.85      2089

    accuracy                           0.80      3000
   macro avg       0.76      0.75      0.76      3000
weighted avg       0.79      0.80      0.80      3000

Confusion Matrix:
 [[ 588  323]
 [ 288 1801]]


Model: SVM
Accuracy: 0.857
Classification Report:
               precision    recall  f1-score   support

           0       0.87      0.62      0.72       911
           1       0.85      0.96      0.90      2089

    accuracy                           0.86      3000
   macro avg       0.86      0.79      0.81      3000
weighted avg       0.86      0.86      0.85      3000

Confusion Matrix:
 [[ 563  348]
 [  81 2008]]
```


# Exportação do Modelo
```python
joblib.dump(best_models['SVM'], 'best_model.pkl')
```
