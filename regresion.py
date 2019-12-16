import sys
print('Python: {}'.format(sys.version))
# scipy
import scipy
print('scipy: {}'.format(scipy.__version__))
# numpy
import numpy
print('numpy: {}'.format(numpy.__version__))
# matplotlib
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
# pandas
import pandas
print('pandas: {}'.format(pandas.__version__))
# scikit-learn
import sklearn
print('sklearn: {}'.format(sklearn.__version__))

import csv
print('csv: {}'.format(csv.__version__))

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import pandas
df = pandas.read_csv('iadata.csv')

import matplotlib.pyplot as plt

import numpy as np

x,y = np.loadtxt('iadata.csv',
                 unpack =True,
                 delimiter = ',')
#creacion de tabla compumo de termperatura mensual
t= np.arange(1,37,1).reshape((36, 1))
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(t, x)

# Make predictions using the testing set
y_pred = regr.predict(t)
plt.scatter(t, x,  color='black')
plt.plot(t,y_pred, color='red', linewidth=3)

#plt.xticks(())
#plt.yticks(())

plt.title('Regresion Lineal Consumo Mensual 2016-2018 Panama', color= 'purple')
plt.xlabel('Meses')
plt.ylabel('Consumo')

plt.show()
##################################################
#creacion de tabla de pronostico
p= np.arange(1,37,1).reshape((36, 1))
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(p, y)

# Make predictions using the testing set
x_pred = regr.predict(p)
plt.scatter(p, y,  color='black')
plt.plot(p,x_pred, color='orange', linewidth=3)

#plt.xticks(())
#plt.yticks(())

plt.title('Regresion Lineal Consumo Mensual 2016-018 Panama', color= 'blue')
plt.xlabel('Meses')
plt.ylabel('Temperatura')

plt.show()


##################################################
w=np.arange(1,37,1).reshape((36, 1))
regr = linear_model.LinearRegression()
regr.fit(p, y)
regr.fit(t, x)
plt.scatter(x, y, color='black')
plt.plot(y_pred,x_pred, color='pink', linewidth=3)

#plt.xticks(())
#plt.yticks(())

plt.title('Regresion Lineal Consumo Mensual 2016-2018 Panama', color= 'blue')
plt.xlabel('Temperatura')
plt.ylabel('Consumo')

plt.show()