import numpy
import pandas
import matplotlib
from basic import Basic
from baseline import Baseline


print("------------------Beginning Training and Testing Phase-----------------")

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

space = (numpy.linspace(1, 101, 100)).astype(int)
scores = []

model = Baseline()
model.preprocess(X, Y)
for k in space:
	print("Epoch: %i", k)
	Y_pred = [model.predict_baseline(i[0], i[1], k) for i in X_test]
	score = numpy.sqrt(numpy.mean((Y_test - Y_pred)**2))
	scores.append(score)

matplotlib.plot(space, scores, 'ro')
plt.show()