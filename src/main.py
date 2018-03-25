import numpy
import pandas
import pickle
import matplotlib.pyplot
from basic import Basic
from baseline import Baseline


print("--------------------Begin Training and Testing Phase-------------------")

try:
	
	with open('../MovieLens 10M Dataset/train.csv', 'r') as a:
		train = pandas.read_csv(a)
		train = train.drop('timestamp', 1)
	a.close()

	X = train.values
	Y = train['rating'].values
	X = numpy.delete(X, 2, axis=1)

	with open('../MovieLens 10M Dataset/test.csv', 'r') as b:
		test = pandas.read_csv(b)
		test = test.drop('timestamp', 1)
	b.close()

	X_test = test.values
	Y_test = test['rating'].values
	X_test = numpy.delete(X_test, 2, axis=1)

except:
	
	print("------------------------Failed To Load Dataset------------------------")
	exit()

else:
	
	print("----------------------------Datasets Loaded----------------------------")


print("-----------------------Baseline Predictor Testing----------------------")

model = Baseline()
model.preprocess(X, Y)
space = (numpy.linspace(1, 301, 50)).astype(int)
scores = []

for k in space:
	print("Epoch: %i", k)
	Y_pred = [model.predict(int(i[0]), int(i[1]), k) for i in X_test]
	score = numpy.sqrt(numpy.mean((Y_test - Y_pred)**2))
	print("RMSE: %f", score)
	scores.append(score)


matplotlib.pyplot.plot(space, scores, 'ro')
matplotlib.pyplot.xlabel('Beta')
matplotlib.pyplot.ylabel('RMSE')
matplotlib.pyplot.title('Baseline Predictor')
matplotlib.pyplot.savefig('../plots/baseline_predictor.png')
matplotlib.pyplot.show()

print("------------------Baseline Predictor Testing Complete------------------")
print("----------------------------All Plots Saved----------------------------")
