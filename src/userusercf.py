from knn import *


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