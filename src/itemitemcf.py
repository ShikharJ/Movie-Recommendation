from knn import *


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
		self.movie_neighbourhood[i] = neighbourhood[:1700]

	def predict(self, u, i):
		b_u = b_i = 0
		if len(self.user_movies[u]):
			b_u = ((self.user_mean_ratings[u] - self.mu) * len(self.user_movies[u])) / (len(self.user_movies[u]) + 120)
		if len(self.movie_users[i]):
			b_i = ((self.movie_mean_ratings[i] - b_u - self.mu) * len(self.movie_users[i])) / (len(self.movie_users[i]) + 120)
		prediction, numer, denom = self.mu + b_u + b_i, 0, 0
		if i not in self.movie_neighbourhood:
			print("User: ", u, " Item: ", i)
			self.generate_neighbourhood(i)
		for k in range(self.N):
			j = self.movie_neighbourhood[i][k][1]
			if self.R[u][j] != 0:
				numer += self.movie_neighbourhood[i][k][0] * (self.R[u][j] - prediction)
				denom += numpy.abs(self.movie_neighbourhood[i][k][0])
		if denom:
			prediction += numer / denom
		return prediction
