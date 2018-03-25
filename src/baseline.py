from basic import *


class Baseline(Basic):

	def __init__(self, beta_u=25, beta_i=25):
		super(Baseline, self).__init__()
		self.beta_u = beta_u
		self.beta_i = beta_i

	def predict(self, u, i):
		b_u = b_i = 0
		if len(self.user_movies[u]):
			b_u = ((self.user_mean_ratings[u] - self.mu) * len(self.user_movies[u])) / (len(self.user_movies[u]) + self.beta_u)
		if len(self.movie_users[i]):
			b_i = ((self.movie_mean_ratings[i] - b_u - self.mu) * len(self.movie_users[i])) / (len(self.movie_users[i]) + self.beta_i)
		return self.mu + b_u + b_i
