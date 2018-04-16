from basic import *
import copy


class RestrictedBoltzmannMachine(Basic):

	def __init__(self, epochs=10, learning_rate=0.001, f=100, k=5):
		super(RestrictedBoltzmannMachine, self).__init__()
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.f = f
		self.k = k
		self.h = numpy.random.rand(self.f) - 0.5

	def set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate

	def set_f(self, f):
		self.f = f

	def set_k(self, k):
		self.k = k

	def weight_initialize(self):
		self.feature_bias = numpy.random.rand(self.f) - 0.5
		self.movie_bias = numpy.random.rand(self.num_movies, self.k) - 0.5
		self.weights = numpy.random.rand(self.f, self.num_movies, self.k) - 0.5

	def train(self):
		for i in range(self.epochs):
			for u in self.users:
				data = copy.deepcopy(self.data[u])
				weights = self.get_weights(self.user_movies[u])
				positive_associations, self.h = self.forward_propagation(data, self.user_movies[u])
				visible_probability = self.backward_propagation(self.h, self.user_movies[u])
				negative_associations, temp = self.forward_propagation(visible_probability, self.user_movies[u])
				weights += self.learning_rate * (positive_associations - negative_associations) / len(self.user_movies[u])
				self.set_weights(self.user_movies[u], w)
				error = numpy.sqrt(numpy.sum((data - visible_probability) ** 2) / len(data))
				print i, u, error

	def get_weights(self, movies):
		A = numpy.zeros((self.f, 1, self.k))
		for m in movies:
			A = numpy.concatenate((a, numpy.expand_dims(self.weights[:, m, :], axis=1)), axis=1)
		return A[:, 1: ,]

	def set_weights(self, movies, weights):
		it = 0
		for m in movies:
			self.weights[:, m, :] += w[:, it, :]
			it += 1

	def get_movie_bias(self, movies):
		A = numpy.zeros((1, self.k))
		for m in movies:
			A = numpy.concatenate((A, numpy.expand_dims(self.movie_bias[m, :], axis=0)), axis=0)
		return A[1:, ]

	def forward_propagation(self, inp, movies):
		hidden_unit = numpy.copy(self.feature_bias)
		for j in range(self.f):
			hidden_unit[j] += numpy.tensordot(inp, self.get_weights(movies)[j])
		hidden_probability = self.sigmoid(hidden_unit)
		hidden_states = hidden_probability > numpy.random.rand(self.f)
		hidden_associations = numpy.zeros((self.f, len(movies), self.k))
		for j in range(self.f):
			hidden_associations[j] = hidden_probability[j] * inp
		return hidden_associations, hidden_states

	def backward_propagation(self, inp, movies):
		visible_unit = self.get_movie_bias(movies)
		for j in range(self.f):
			visible_unit += inp[j] * self.get_weights(movies)[j]
		visible_probability = self.sigmoid(visible_unit)
		return visible_probability

	def predict(self, u, i):
		w = self.get_weights(self.user_movies[u])
		data = copy.deepcopy(self.data[u])
		probabilities = numpy.ones(5)
		mx, index = -1, 0
		for i in range(5):
			probability = 1.0
			for j in range(self.f):
				temp = 1.0 + numpy.exp(numpy.tensordot(data, self.get_weights(self.user_movies[u])[j]) + self.feature_bias[j])
				probability *= temp
			probabilities[i] = probability
			if mx < probabilities[i]:
				index = i
				mx = probabilities[i]
		return index

	def sigmoid(self, x):
		return 1 / (1 + numpy.exp(-x))
