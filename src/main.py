import numpy
import pandas
import time
import matplotlib.pyplot
from basic import Basic
from baseline import Baseline
from userusercf import UserUserCollaborativeFiltering
from itemitemcf import ItemItemCollaborativeFiltering
from svd import SingularValueDecomposition


print("--------------------Begin Training and Testing Phase-------------------")

try:
	
	with open('../MovieLens 10M Dataset/train.csv', 'r') as a:
		train = pandas.read_csv(a)
		train = train.drop('timestamp', 1)
	a.close()

	X = train.values
	Y = train['rating'].values
	X = numpy.delete(X, 2, axis=1)
	X = X.astype(int)

	with open('../MovieLens 10M Dataset/test.csv', 'r') as b:
		test = pandas.read_csv(b)
		test = test.drop('timestamp', 1)
	b.close()

	X_test = test.values
	Y_test = test['rating'].values
	X_test = numpy.delete(X_test, 2, axis=1)
	X_test = X_test.astype(int)

except Exception as ex:

	print(ex)
	print("------------------------Failed To Load Dataset------------------------")
	exit()

else:
	
	print("----------------------------Datasets Loaded----------------------------")

'''
print("-----------------------Baseline Predictor Testing----------------------")

space = (numpy.linspace(1, 300, 100)).astype(int)
scores = []

model = Baseline()
model.train(X, Y)

for k in space:
	print("Epoch: ", k)
	model.set_beta_u(k)
	model.set_beta_i(k)
	score = model.RMSE(X_test, Y_test)
	print("RMSE: ", score)
	scores.append(score)

matplotlib.pyplot.plot(space, scores, 'ro')
matplotlib.pyplot.xlabel('Beta')
matplotlib.pyplot.ylabel('RMSE')
matplotlib.pyplot.title('Baseline Predictor')
matplotlib.pyplot.savefig('../plots/baseline_predictor.png')
matplotlib.pyplot.show()

print("------------------Baseline Predictor Testing Complete------------------")


print("----------------User User Collaborative Filtering Testing--------------")

model = UserUserCollaborativeFiltering()
model.train(X, Y)
space = (numpy.linspace(1, 2000, 100)).astype(int)
scores = []
times = []

for k in space:
	print("Epoch: %i", k)
	model.set_neighbourhood(k)
	if k > 1:
		start_time = time.time()
	score = model.RMSE(X_test, Y_test)
	if k > 1:
		times.append(time.time() - start_time)
		print("Time: ", time.time() - start_time)
	print("RMSE: ", score)
	scores.append(score)

matplotlib.pyplot.plot(space, scores, '+')
matplotlib.pyplot.xlabel('Nearest Neighbours')
matplotlib.pyplot.ylabel('RMSE')
matplotlib.pyplot.title('User - User Collaborative Filtering')
matplotlib.pyplot.savefig('../plots/user_user_collaborative_filtering1.png')
matplotlib.pyplot.show()
matplotlib.pyplot.gcf().clear()

matplotlib.pyplot.plot(space[1:], times, 'k^:')
matplotlib.pyplot.xlabel('Nearest Neighbours')
matplotlib.pyplot.ylabel('Time (in seconds)')
matplotlib.pyplot.title('User - User Collaborative Filtering')
matplotlib.pyplot.savefig('../plots/user_user_collaborative_filtering2.png')

print("-----------User User Collaborative Filtering Testing Complete----------")


print("----------------Item Item Collaborative Filtering Testing--------------")

model = ItemItemCollaborativeFiltering()
model.train(X, Y)
space = (numpy.linspace(1, 2000, 100)).astype(int)
scores = []
times = []

for k in space:
	print("Epoch: %i", k)
	model.set_neighbourhood(k)
	if k > 1:
		start_time = time.time()
	score = model.RMSE(X_test, Y_test)
	if k > 1:
		times.append(time.time() - start_time)
		print("Time: ", time.time() - start_time)
	print("RMSE: ", score)
	scores.append(score)

matplotlib.pyplot.plot(space, scores, 'g+')
matplotlib.pyplot.xlabel('Nearest Neighbours')
matplotlib.pyplot.ylabel('RMSE')
matplotlib.pyplot.title('Item - Item Collaborative Filtering')
matplotlib.pyplot.savefig('../plots/item_item_collaborative_filtering1.png')
matplotlib.pyplot.show()
matplotlib.pyplot.gcf().clear()

matplotlib.pyplot.plot(space[1:], times, 'k^:')
matplotlib.pyplot.xlabel('Nearest Neighbours')
matplotlib.pyplot.ylabel('Time (in seconds)')
matplotlib.pyplot.title('Item - Item Collaborative Filtering')
matplotlib.pyplot.savefig('../plots/item_item_collaborative_filtering2.png')
matplotlib.pyplot.show()

print("-----------Item Item Collaborative Filtering Testing Complete----------")


print("------------------Singular Value Decomposition Testing-----------------")

model = SingularValueDecomposition()
model.preprocess(X, Y)
space = (numpy.linspace(0.001, 0.01, 50))
max_error = 100000.0
best_learning_rate = 0.001
scores1 = []
scores2 = []
scores3 = []

for k in space:
	print("Epoch: %i", k)
	model.set_learning_rate(k)
	score = model.train(X, Y, X_test, Y_test)
	print("RMSE: ", score)
	if score < max_error:
		max_error = score
		best_learning_rate = k
	scores1.append(score)

matplotlib.pyplot.plot(space, scores1, '+')
matplotlib.pyplot.xlabel('Learning Rate')
matplotlib.pyplot.ylabel('RMSE')
matplotlib.pyplot.title('Singular Value Decomposition')
matplotlib.pyplot.savefig('../plots/singular_value_decomposition1.png')
matplotlib.pyplot.show()
matplotlib.pyplot.gcf().clear()

space = (numpy.linspace(1, 600, 30)).astype(int)
model.set_learning_rate(best_learning_rate)
max_error = 100000.0
best_f = 1

for k in space:
	print("Epoch: %i", k)
	model.set_f(k)
	score = model.RMSE(X, Y, X_test, Y_test)
	print("RMSE: ", score)
	if score < max_error:
		max_error = score
		best_f = k
	scores2.append(score)

matplotlib.pyplot.plot(space, scores2, 'k^:')
matplotlib.pyplot.xlabel('Number of Features')
matplotlib.pyplot.ylabel('RMSE')
matplotlib.pyplot.title('Singular Value Decomposition')
matplotlib.pyplot.savefig('../plots/singular_value_decomposition2.png')
matplotlib.pyplot.show()
matplotlib.pyplot.gcf().clear()

space = (numpy.linspace(0.001, 1, 1000))
model.set_f(best_f)
model.set_bias(True)

for k in space:
	print("Epoch: %i", k)
	model.set_k_u(k)
	model.set_k_b(k)
	model.set_k_m(k)
	score = model.RMSE(X, Y, X_test, Y_test)
	print("RMSE: ", score)
	scores3.append(score)

matplotlib.pyplot.plot(space, scores3, 'x')
matplotlib.pyplot.xlabel('Bias')
matplotlib.pyplot.ylabel('RMSE')
matplotlib.pyplot.title('Singular Value Decomposition')
matplotlib.pyplot.savefig('../plots/singular_value_decomposition3.png')
matplotlib.pyplot.show()

print("-------------Singular Value Decomposition Testing Complete-------------")
'''

print("----------------------------All Plots Saved----------------------------")
