import numpy
import pandas
import time
import matplotlib.pyplot
from basic import Basic
from baseline import Baseline
from userusercf import UserUserCollaborativeFiltering
from itemitemcf import ItemItemCollaborativeFiltering
from svd import SingularValueDecomposition
from rbm import RestrictedBoltzmannMachine


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
matplotlib.pyplot.gcf().clear()

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
matplotlib.pyplot.gcf().clear()

matplotlib.pyplot.plot(space[1:], times, 'k^:')
matplotlib.pyplot.xlabel('Nearest Neighbours')
matplotlib.pyplot.ylabel('Time (in seconds)')
matplotlib.pyplot.title('User - User Collaborative Filtering')
matplotlib.pyplot.savefig('../plots/user_user_collaborative_filtering2.png')
matplotlib.pyplot.gcf().clear()

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
matplotlib.pyplot.gcf().clear()

matplotlib.pyplot.plot(space[1:], times, 'k^:')
matplotlib.pyplot.xlabel('Nearest Neighbours')
matplotlib.pyplot.ylabel('Time (in seconds)')
matplotlib.pyplot.title('Item - Item Collaborative Filtering')
matplotlib.pyplot.savefig('../plots/item_item_collaborative_filtering2.png')
matplotlib.pyplot.gcf().clear()

print("-----------Item Item Collaborative Filtering Testing Complete----------")


print("------------------Singular Value Decomposition Testing-----------------")

Y = Y.reshape(Y.shape[0], 1)
X_new = numpy.concatenate((X, Y), axis=1)
numpy.random.shuffle(X_new)
limit = (int)(X_new.shape[0] * 9 / 10)
X, X_val = X_new[:limit, :], X_new[limit:, :]
Y = X[:, 2]
Y_val = X_val[:, 2]
X = numpy.delete(X, 2, axis=1)
X = X.astype(int)
X_val = numpy.delete(X_val, 2, axis=1)
X_val = X_val.astype(int)
del X_new

model = SingularValueDecomposition()
model.preprocess(X, Y)
space = (numpy.linspace(0.001, 0.01, 10))
max_error = 100000.0
best_learning_rate = 0.001
scores1 = []
scores2 = []
scores3 = []

for k in space:
	print("Epoch: %i", k)
	model.set_learning_rate(k)
	score = model.train(X, Y, X_val, Y_val)
	print("RMSE: ", score)
	if score < max_error:
		max_error = score
		best_learning_rate = k
	scores1.append(score)

matplotlib.pyplot.plot(space, scores1, '+-')
matplotlib.pyplot.xlabel('Learning Rate')
matplotlib.pyplot.ylabel('Cross Validation Error')
matplotlib.pyplot.title('Singular Value Decomposition')
matplotlib.pyplot.savefig('../plots/singular_value_decomposition1.png')
matplotlib.pyplot.gcf().clear()

space = (numpy.linspace(1, 600, 10)).astype(int)
model.set_learning_rate(best_learning_rate)
max_error = 100000.0
best_f = 1

for k in space:
	print("Epoch: %i", k)
	model.set_f(k)
	score = model.train(X, Y, X_val, Y_val)
	print("RMSE: ", score)
	if score < max_error:
		max_error = score
		best_f = k
	scores2.append(score)

matplotlib.pyplot.plot(space, scores2, 'k^:')
matplotlib.pyplot.xlabel('Number of Features')
matplotlib.pyplot.ylabel('Cross Validation Error')
matplotlib.pyplot.title('Singular Value Decomposition')
matplotlib.pyplot.savefig('../plots/singular_value_decomposition2.png')
matplotlib.pyplot.gcf().clear()

space = (numpy.linspace(0.01, 1, 10))
model.set_f(best_f)
model.set_bias(True)

for k in space:
	print("Epoch: %i", k)
	model.set_k_u(k)
	model.set_k_b(k)
	model.set_k_m(k)
	score = model.train(X, Y, X_val, Y_val)
	score = model.RMSE(X_test, Y_test)
	print("RMSE: ", score)
	scores3.append(score)

matplotlib.pyplot.plot(space, scores3, 'rx-')
matplotlib.pyplot.xlabel('Bias')
matplotlib.pyplot.ylabel('RMSE')
matplotlib.pyplot.title('Singular Value Decomposition')
matplotlib.pyplot.savefig('../plots/singular_value_decomposition3.png')
matplotlib.pyplot.gcf().clear()

print("-------------Singular Value Decomposition Testing Complete-------------")


print("------------------Restricted Boltzmann Machine Testing-----------------")

model = RestrictedBoltzmannMachine()
model.preprocess(X, Y)
model.weight_initialize()
space = (numpy.linspace(0.001, 0.01, 10))
max_error = 100000.0
best_learning_rate = 0.001
scores1 = []
scores2 = []
scores3 = []

for k in space:
	print("Epoch: %i", k)
	model.set_learning_rate(k)
	model.train()
	score = model.RMSE(X_val, Y_val)
	print("RMSE: ", score)
	if score < max_error:
		max_error = score
		best_learning_rate = k
	scores1.append(score)

matplotlib.pyplot.plot(space, scores1, '+-')
matplotlib.pyplot.xlabel('Learning Rate')
matplotlib.pyplot.ylabel('Cross Validation Error')
matplotlib.pyplot.title('Restricted Boltzmann Machine')
matplotlib.pyplot.savefig('../plots/restricted_boltzmann_machine1.png')
matplotlib.pyplot.gcf().clear()

space = (numpy.linspace(1, 600, 10)).astype(int)
model.set_learning_rate(best_learning_rate)
max_error = 100000.0
best_f = 1

for k in space:
	print("Epoch: %i", k)
	model.set_f(k)
	model.train()
	score = model.RMSE(X_val, Y_val)
	print("RMSE: ", score)
	if score < max_error:
		max_error = score
		best_f = k
	scores2.append(score)

matplotlib.pyplot.plot(space, scores2, 'k^:')
matplotlib.pyplot.xlabel('Number of Features')
matplotlib.pyplot.ylabel('Cross Validation Error')
matplotlib.pyplot.title('Restricted Boltzmann Machine')
matplotlib.pyplot.savefig('../plots/restricted_boltzmann_machine2.png')
matplotlib.pyplot.gcf().clear()

space = (numpy.linspace(0.01, 1, 10))
model.set_f(best_f)

for k in space:
	print("Epoch: %i", k)
	model.set_k(k)
	model.train()
	score = model.RMSE(X_val, Y_val)
	print("RMSE: ", score)
	scores3.append(score)

matplotlib.pyplot.plot(space, scores3, 'rx-')
matplotlib.pyplot.xlabel('Bias')
matplotlib.pyplot.ylabel('RMSE')
matplotlib.pyplot.title('Restricted Boltzmann Machine')
matplotlib.pyplot.savefig('../plots/restricted_boltzmann_machine3.png')
matplotlib.pyplot.gcf().clear()

print("-------------Restricted Boltzmann Machine Testing Complete-------------")


print("----------------------Hybrid Combinational Testing---------------------")

model1 = Baseline()
model1.train(X, Y)
model1.set_beta_u(120)
model1.set_beta_i(120)
model1.RMSE(X_test, Y_test, save=True)

model2 = UserUserCollaborativeFiltering()
model2.train(X, Y)
model2.set_neighbourhood(NUM)
model2.RMSE(X_test, Y_test, save=True)

model3 = ItemItemCollaborativeFiltering()
model3.train(X, Y)
model3.set_neighbourhood(2000)
model3.RMSE(X_test, Y_test, save=True)

Y = Y.reshape(Y.shape[0], 1)
X_new = numpy.concatenate((X, Y), axis=1)
numpy.random.shuffle(X_new)
limit = (int)(X_new.shape[0] * 9 / 10)
X, X_val = X_new[:limit, :], X_new[limit:, :]
Y = X[:, 2]
Y_val = X_val[:, 2]
X = numpy.delete(X, 2, axis=1)
X = X.astype(int)
X_val = numpy.delete(X_val, 2, axis=1)
X_val = X_val.astype(int)
del X_new

model4 = SingularValueDecomposition()
model4.preprocess(X, Y)
model4.set_learning_rate(0.001)
model4.set_f(533)
model4.set_k_u(0.1)
model4.set_k_m(0.1)
model4.set_k_b(0.1)
model4.train(X, Y, X_val, Y_val)
model4.RMSE(X_test, Y_test, save=True)

model5 = RestrictedBoltzmannMachine()
model5.preprocess(X, Y)
model5.weight_initialize()
model5.set_learning_rate(0.001)
model5.set_f(100)
model5.set_k(5)
model5.train()
model5.RMSE(X_test, Y_test, save=True)

X_new = numpy.concatenate((model1.Y_pred, model2.Y_pred, model3.Y_pred, model4.Y_pred, model5.Y_pred), axis=1)
m, c = numpy.linalg.lstsq(X_new, Y_test)

print("-----------------Hybrid Combinational Testing Complete-----------------")

print("----------------------------All Plots Saved----------------------------")
