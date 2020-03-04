import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)
data = imagenes.reshape((n_imagenes, -1)) 
data_train, data_val, y_train, y_val = train_test_split(data, target, train_size=0.8)
x_train, x_test, y_train, y_test = train_test_split(data_train, y_train, train_size=0.5)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
data_val = scaler.fit_transform(data_val)

cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]
x_train = x_train @ vectores
x_test = x_test @ vectores
data_val = data_val @ vectores

c = np.logspace(1,10,100)
score = np.zeros(len(c))
count = 0
for j in c:
    svc = SVC(C=j, kernel = 'linear')
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    score[count] = f1_score(y_test, y_pred, average = 'macro', labels = np.arange(10))
    count = count + 1



cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(7,7))
plt.imshow(cm, interpolation='nearest')
tick_marks = np.arange(9)
classNames = ['0','1','2','3','4','5','6','7','8','9']
#plt.title('TRAIN', fontsize=15)
plt.ylabel('True', fontsize=15)
plt.xlabel('Predict', fontsize=15)
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames, rotation=45)
plt.savefig('matrix_confusion.png')
plt.show()