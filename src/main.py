import pandas
import numpy
import matplotlib


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

	train = preprocess

	def RMSE(self, X, Y):
		return None

	def MAE(self, X, Y):
		return None


class Baseline(Basic):

	def __init__(self, beta_u=25, beta_i=25):
		super(Baseline, self).__init__()
		self.beta_u = beta_u
		self.beta_i = beta_i

	def predict(self, u, i):
		if len(self.user_movies[u]):
			b_u = ((self.user_mean_ratings[u] - self.mu) * len(self.user_movies[u])) / (len(self.user_movies[u]) + self.beta_u)
		if len(self.movie_users[i]):
			b_i = ((self.movie_mean_ratings[u] - b_u - self.mu) * len(self.movie_users[i])) / (len(self.movie_users[i]) + self.beta_i)
		return self.mu + b_u + b_i


class KNearestNeighbours(Basic):

	def __init__(self, N):
		super(KNearestNeighbours, self).__init__()
		self.N = N

	def pearson_correlation(self, u, v):
		common_movies = numpy.intersect1d(self.user_movies[u], self.user_movies[v], assume_unique=True)
		if (not len(common_movies)):
			l1, l2 = [0], [0]
		else:
			l1 = [self.R[u][i] - self.user_mean_ratings[u] for i in common_movies]
			l2 = [self.R[v][i] - self.user_mean_ratings[v] for i in common_movies]
		r = numpy.dot(l1, l2)
		if r:
			r /= numpy.sqrt(numpy.sum(numpy.square(l1)) * numpy.sum(numpy.square(l2)))
		return r

	def cosine_correlation(self, i, j):
		common_users = numpy.intersect1d(self.movie_users[i], self.movie_users[j], assume_unique=True)
		if (not len(common_users)):
			l1, l2 = [0], [0]
		else:
			l1 = [self.R[u][i] - self.user_mean_ratings[u] for u in common_users]
			l2 = [self.R[u][j] - self.user_mean_ratings[u] for u in common_users]
		r = numpy.dot(l1, l2)
		if r:
			r /= numpy.sqrt(numpy.sum(numpy.square(l1)) * numpy.sum(numpy.square(l2)))
		return r


class UserUserCollaborativeFiltering(KNearestNeighbours):
	
	def __init__(self, N=20):
		super(UserUserCollaborativeFiltering, self).__init__(N)
		self.user_neighbourhood = {}

	def generate_neighbourhood(self, u):
		neighbourhood = []
		for v in self.users:
			if v != u:
				neighbourhood.append((self.pearson_correlation(u, v), v))
		neighbourhood = sorted(neighbourhood, reverse=True)
		return neighbourhood

	def predict(self, u, i):
		prediction, numer, denom = self.user_mean_ratings[u], 0,  0
		if not self.user_neighbourhood.has_key[u]:
			self.user_neighbourhood[u] = self.generate_neighbourhood(u)
		for n in range(self.N):
			v = self.user_neighbourhood[u][n][1]
			if self.R[v][i] != 0:
				if self.user_standard_deviations[v] != 0:
					numer += self.user_neighbourhood[u][n][0] * (self.R[v][i] - self.user_mean_ratings[v]) / self.user_standard_deviations[v]
				denom += numpy.abs(self.user_neighbourhood[u][n][0])
		if b != 0:
			prediction += self.user_standard_deviations[u] * numer / denom
		return prediction


class ItemItemCollaborativeFiltering(KNearestNeighbours):
	
	def __init__(self, N=20):
		super(ItemItemCollaborativeFiltering, self).__init__(N)
		self.movie_neighbourhood = {}

	def generate_neighbourhood(self, i):
		neighbourhood = []
		for j in self.movies:
			if j != i:
				neighbourhood.append((self.cosine_correlation(i, j), j))
		neighbourhood = sorted(neighbourhood, reverse=True)
		return neighbourhood

	def predict(self, u, i):
		return None


class SingularValueDecomposition(Basic):

	def __init__(self):
		super(SingularValueDecomposition, self).__init__()

	def predict(self, u, i):
		return None


class RestrictedBoltzmannMachines(Basic):

	def __init__(self):
		super(RestrictedBoltzmannMachines, self).__init__()

	def predict(self, u, i):
		return None