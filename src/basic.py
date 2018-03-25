import numpy


class Basic(object):

	def __init__(self):
		self.user_movies = {}
		self.movie_users = {}
		self.movie_mean_ratings = {}
		self.user_mean_ratings = {}
		self.user_standard_deviations = {}
		self.R = 0
		self.users = 0
		self.movies = 0
		self.mu = 0
		self.num_users = 0
		self.num_movies = 0

	def preprocess(self, X, Y):

		self.users = numpy.unique(X[:, 0])
		self.movies = numpy.unique(X[:, 1])
		self.num_users = int(X[:, 0].max())
		self.num_movies = int(X[:, 1].max())
		self.R = numpy.zeros(shape=(self.num_users + 1, self.num_movies + 1))
		
		for i, x in enumerate(X):
			self.R[int(x[0])][int(x[1])] = Y[i]

		for u in range(self.num_users + 1):
			self.user_movies[u] = list(filter(lambda x: self.R[u][int(x)] > 0, self.movies))
			if len(self.user_movies[u]):
				l = [self.R[u][int(i)] for i in self.user_movies[u]]
			else:
				l = 0
			self.user_mean_ratings[u] = numpy.mean(l)
			self.user_standard_deviations[u] = numpy.std(l)

		for i in range(self.num_movies + 1):
			self.movie_users[i] = list(filter(lambda x: self.R[int(x)][i] > 0, self.users))
			if len(self.movie_users[i]):
				self.movie_mean_ratings[i] = numpy.mean([self.R[int(u)][i] for u in self.movie_users[i]])
			else:
				self.movie_mean_ratings[i] = 0

		self.mu = numpy.mean(Y)

	train = preprocess

	def RMSE(self, X, Y):
		Y_pred = [self.predict(a[0], a[1]) for a in X]
		RMSE = numpy.sqrt(numpy.mean((Y - Y_pred) ** 2))
		return RMSE

	def MAE(self, X, Y):
		Y_pred = [self.predict(a[0], a[1]) for a in X]
		MAE = numpy.mean(numpy.abs(Y - Y_pred))
		return MAE
