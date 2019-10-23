# https://drive.google.com/open?id=1ONmD-kYBHIiwJYRDMFKZy2g5-ZQwIwF7
import numpy as np
import pandas as pd
from scipy import stats
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
#import seaborn as sns

df = pd.read_csv('fandango_score_comparison.csv')
print(df.head())

feature_cols = ['Fandango_Stars', 'RT_user_norm', 'RT_norm',
'Metacritic_user_norm', 'Metacritic_norm']
X = df.loc[:, feature_cols]

df.rename(columns={'Metacritic_user_nom':'Metacritic_user_norm'},
inplace=True)

rankings_lst = ['Fandango_Stars',
'RT_user_norm',
'RT_norm',
'IMDB_norm',
'Metacritic_user_norm',
'Metacritic_norm']

y = df['IMDB_norm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=43)

dim = len(feature_cols)
dim += 1

X_train = X_train.assign( independent = pd.Series([1] * len(y_train), index=X_train.index))
X_test = X_test.assign( independent = pd.Series([1] * len(y_train), index=X_test.index))

P_train = X_train.as_matrix(columns=None)
P_test = X_test.as_matrix(columns=None)
q_train = np.array(y_train.values).reshape(-1,1)
q_test = np.array(y_test.values).reshape(-1,1)

P = tf.placeholder(tf.float32,[None,dim])
q = tf.placeholder(tf.float32,[None,1])
T = tf.Variable(tf.ones([dim,1]))

bias = tf.Variable(tf.constant(1.0, shape = [n_dim]))
q_ = tf.add(tf.matmul(P, T),bias)

cost = tf.reduce_mean(tf.square(q_ - q))
learning_rate = 0.0001
training_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init_op = tf.global_variables_initializer()
cost_history = np.empty(shape=[1],dtype=float)

