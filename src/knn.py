from basic import *


class KNearestNeighbours(Basic):

	def __init__(self, N):
		super(KNearestNeighbours, self).__init__()
		self.N = N

	def set_neighbourhood(self, N):
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
