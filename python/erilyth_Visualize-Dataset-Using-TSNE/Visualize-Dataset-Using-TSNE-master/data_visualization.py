import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE

# visulaize the important characteristics of the dataset
import matplotlib.pyplot as plt

def get_data(file_name, label_name, test_percent, drop_rows = 0):
	dataframe_all = pd.read_csv(file_name)
	if drop_rows == 1:
		# Drop rows which don't have values specified for all the columns
		dataframe_all = dataframe_all.dropna(axis=0, how='any')
	num_rows = dataframe_all.shape[0]
	# step 2: remove useless data
	# count the number of missing elements (NaN) in each column
	counter_nan = dataframe_all.isnull().sum()
	counter_without_nan = counter_nan[counter_nan==0]
	# remove the columns with missing elements
	dataframe_all = dataframe_all[counter_without_nan.keys()]
	# remove the first 7 columns which contain no discriminative information
	dataframe_all = dataframe_all.ix[:,:]
	# the list of columns (the last column is the class label)
	columns = dataframe_all.columns
	print columns
	# step 3: get class labels y and then encode it into number 
	# get class label data
	y = dataframe_all[[label_name]].values
	y = y.reshape([y.shape[0]])
	# print y.shape
	# encode the class label
	class_labels = np.unique(y)
	label_encoder = LabelEncoder()
	y = label_encoder.fit_transform(y)
	# step 4: get features (x) and scale the features
	# get x and convert it to numpy array
	# print dataframe_all
	dataframe_all = dataframe_all.drop(label_name, 1)
	dataframe_all = dataframe_all.apply(LabelEncoder().fit_transform)
	x = dataframe_all.ix[:,:].values
	standard_scaler = StandardScaler()
	x_std = standard_scaler.fit_transform(x)
	# step 5: split the data into training set and test set
	x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size = test_percent, random_state = 0)
	return x_train, x_test, y_train, y_test

def perform_tsne(x_test, y_test, title):
	# t-distributed Stochastic Neighbor Embedding (t-SNE) visualization
	tsne = TSNE(n_components=2, random_state=0)
	x_test_2d = tsne.fit_transform(x_test)

	# scatter plot the sample points among 5 classes
	markers=('s', 'd', 'o', '^', 'v')
	color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple', 4:'cyan'}
	plt.figure()
	for idx, cl in enumerate(np.unique(y_test)):
	    plt.scatter(x=x_test_2d[y_test==cl,0], y=x_test_2d[y_test==cl,1], c=color_map[idx], marker=markers[idx], label=cl)
	plt.xlabel('X in t-SNE')
	plt.ylabel('Y in t-SNE')
	plt.legend(loc='upper left')
	plt.title('t-SNE visualization for ' + title)
	plt.show()
