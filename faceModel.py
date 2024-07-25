
import csv # for reading the csv file
import pandas as pd # uses dataframes from pandas
import numpy as np # used for arrays
from sklearn.ensemble import RandomForestClassifier # used for creating FSS score predictors --> Random Forest algorithm
from sklearn.svm import SVC
from sklearn.model_selection import KFold # for implementing k-folds
import matplotlib.pyplot as plt # used for plotting the graphs and heatmap
from sklearn import tree
import seaborn as sns # for creating the confusion matrix heatmap


# FUNCTIONS:

def create_conf_matrix(y_preds, y_reals):
	TP = 0
	FN = 0
	FP = 0
	TN = 0
	falsePos = []
	falseNeg = []

	for i in range(len(y_preds)):
		pred_val = y_preds[i]
		test_val = y_reals[i]

		if test_val == 1 and pred_val == 1:
			TP += 1
		if test_val == 1 and pred_val == 0:
			FN += 1
		if test_val == 0 and pred_val == 1:
			FP += 1
		if test_val == 0 and pred_val == 0:
			TN += 1

	return [TP, FN, FP, TN]

# calculates accuracy of model results
def accuracy(matrix):
	TP = matrix[0]
	TN = matrix[3]
	acc = (TP + TN) / (matrix[0] + matrix[1] + matrix[2] + matrix[3])
	return acc

# calculates the sensitivity of model results
def sensitivity(matrix):
	TP = matrix[0]
	FN = matrix[1]
	sens = TP / (TP + FN)
	return sens

# calculates the specificity of model results
def specificity(matrix):
	TN = matrix[3]
	FP = matrix[2]
	spec = TN / (TN + FP)
	return spec


# MAIN

# dataset = np.array(pd.read_csv('ModelData.csv'))
dataset = pd.read_csv('ModelData.csv') # dataset of confidence levels
dataset = dataset.replace(np.nan, 0)
dataset = np.array(dataset)
target_data = dataset[:, 0] # UPDATE!
dataset = dataset[:, [1, 2, 3]]
num_entries = len(dataset)

labels = ["Thermal", "OpenFace", "Kinect"] # UPDATE COLUMN TITLES
classes = ["0", "1"]

kfold = KFold(n_splits = 10, shuffle = True, random_state = 100) # creates 10-fold splits
face_preds = []
face_truths = []

for train_index, test_index in kfold.split(dataset):

	x_train, x_test = dataset[train_index], dataset[test_index]
	y_train, y_test = target_data[train_index], target_data[test_index]

	
	# classifier = RandomForestClassifier(max_depth = 6, min_samples_leaf = 3, min_samples_split = 10, class_weight = 'balanced', bootstrap = False, max_features = "auto", random_state = 100) # generates random forest model
	classifier = RandomForestClassifier(max_depth = 3, min_samples_leaf = 10, min_samples_split = 30, class_weight = 'balanced', bootstrap = False, max_features = "auto", random_state = 100) # generates random forest model
	classifier.fit(x_train, y_train) # trains the model using the training sets
	y_pred = classifier.predict(x_test) # predicts the data for the test dataset
	
	for i in range(len(y_pred)):
		face_preds.append(y_pred[i])
		face_truths.append(y_test[i])


matrix = create_conf_matrix(face_preds, face_truths)
print("[TP, FN, FP, TN] = ", end = "")
print(matrix)
print("\nAccuracy: " + str(accuracy(matrix)))
print("Sensitivity: " + str(sensitivity(matrix)))
print("Specificity: " + str(specificity(matrix)) + "\n")

# plot a decision tree
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 8), dpi = 400)
tree.plot_tree(classifier.estimators_[0], feature_names = labels, class_names = classes, filled = True, impurity = False, precision = 1, fontsize = 10)
fig.savefig('decisionTree.png')

# create confusion matrix heatmap

# cf_matrix = np.array([[286, 7], [1, 662]])
# group_names = ['True Positive','False Positive','False Negative','True Negative']
# group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
# group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
# labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
# labels = np.asarray(labels).reshape(2,2)

# ax = sns.heatmap(cf_matrix, annot = labels, fmt = '', cmap = 'Blues', square = True)
# ax.set_xlabel('True Human')
# ax.set_ylabel('Predicted Human')
# ax.set_title('Confusion Matrix Heatmap for the Random Forest Model Predictions')
# axis_labels = ["Positive", "Negative"]
# ax.set_xticks([0.5, 1.5])
# ax.set_yticks([0.5, 1.5])
# ax.set_xticklabels(axis_labels)
# ax.set_yticklabels(axis_labels)
# plt.savefig('ConfMatrix.png')

