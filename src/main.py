import pandas
import numpy
import matplotlib

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
		if len(self.user_movies[u]):
			b_u = ((self.user_mean_ratings[u] - self.mu) * len(self.user_movies[u])) / (len(self.user_movies[u]) + 25)
		if len(self.movie_users[i]):
			b_i = ((self.movie_mean_ratings[u] - b_u - self.mu) * len(self.movie_users[i])) / (len(self.movie_users[i]) + 25)
		prediction, numer, denom = self.mu + b_u + b_i, 0, 0
		if not self.movie_neighbourhood.has_key[i]:
			self.movie_neighbourhood[i] = self.generate_neighbourhood(i)
		for k in range(self.N):
			j = self.movie_neighbourhood[i][k][1]
			if self.R[u][j] != 0:
				a += self.movie_neighbourhood[i][k][0] * (self.R[u][j] - prediction)
				b += np.abs(self.movie_neighbourhood[i][k][0])
		if b:
			prediction += a / b
		return prediction


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