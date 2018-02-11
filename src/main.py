import pandas
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
		self.num_users = X[:, 0].max()
		self.num_movies = X[:, 1].max()
		self.R = numpy.zeros(shape=(self.num_users + 1, self.num_movies + 1))
		
		for i, x in enumerate(X):
			self.R[x[0]][x[1]] = Y[i]

		for u in range(self.num_users + 1):
			self.user_movies[i] = filter(lambda x: self.R[u][x] > 0, self.movies)
			if len(self.user_movies[u]):
				l = [self.R[u][i] for i in self.user_movies[u]]
			else:
				l = 0
			self.user_mean_ratings[u] = numpy.mean(l)
			self.user_standard_deviations[u] = numpy.std(l)

		for i in range(self.num_movies + 1):
			self.movie_users[i] = filter(lambda x: self.R[x][i] > 0, self.users)
			if len(self.movie_users[i]):
				self.movie_mean_ratings[i] = numpy.mean([self.R[u][i] for u in self.movie_users[i]])
			else:
				self.movie_mean_ratings[i] = 0

		self.mu = numpy.mean(Y)


class Baseline(Basic):

	def __init__(self, beta_u=25, beta_i=25):
		super(Baseline, self).__init__()
		self.beta_u = beta_u
		self.beta_i = beta_i

	def train(self, X, Y):
		Basic.preprocess(self, X, Y)

	def predict(self, u, i):
		if len(self.user_movies[u]):
			b_u = ((self.user_mean_ratings[u] - self.mu) * len(self.user_movies[u])) / (len(self.user_movies[u]) + self.beta_u)
		if len(self.movie_users[i]):
			b_i = ((self.movie_mean_ratings[u] - b_u - self.mu) * len(self.movie_users[i])) / (len(self.movie_users[i]) + self.beta_i)
		return self.mu + b_u + b_i


class CollaborativeFiltering(Basic):
	
	def __init__(self, beta_u=25, beta_i=25):
		super(Baseline, self).__init__()
		self.beta_u = beta_u
		self.beta_i = beta_i

	def train(self, X, Y):
		Basic.preprocess(self, X, Y)	
