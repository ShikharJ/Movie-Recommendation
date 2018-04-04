from basic import *


class SingularValueDecomposition(Basic):

	def __init__(self, epochs=100, f=60, learning_rate=0.007, k_u=0.05, k_m=0.05, k_b=0.05, bias=False):
		super(SingularValueDecomposition, self).__init__()
		self.epochs = epochs
		self.f = f
		self.learning_rate = learning_rate
		self.k_u = k_u
		self.k_m = k_m
		self.k_b = k_b
		self.bias = bias

	def set_learning_rate(self, learning_rate):
		self.learning_rate = learning_rate

	def set_f(self, f):
		self.f = f

	def set_k_u(self, k_u):
		self.k_u = k_u

	def set_k_m(self, k_m):
		self.k_m = k_m

	def set_k_b(self, k_b):
		self.k_b = k_b

	def set_bias(self, bias):
		self.bias = bias

	def predict(self, u, i):
		x = numpy.dot(self.U[u], self.M[i])
		if self.bias == True:
			x += self.mu + self.alpha[u] + self.beta[i]
		return x

	def train(self, X, Y, X_val, Y_val):
		self.U = numpy.empty((self.num_users + 1, self.f))
		self.M = numpy.empty((self.num_movies + 1, self.f))
		self.alpha = numpy.zeros(self.num_users + 1)
		self.beta = numpy.zeros(self.num_movies + 1)
		initial_val = numpy.random.uniform(-0.01, 0.01)
		for row in self.U:
			row.fill(initial_val)
		for row in self.M:
			row.fill(initial_val)
		cost = float('inf')
		U_temp = numpy.zeros(self.f)
		for r in range(self.epochs):
			for x in X:
				u, m = x[0], x[1]
				error = self.R[u][m] - self.predict(u, m)
				U_temp = self.U[u] + self.learning_rate * (error * self.M[m] - self.U[u] * self.k_u)
				self.M[m] += self.learning_rate * (error * self.U[u] - self.M[m] * self.k_m)
				self.U[u] = U_temp[:]
				if self.bias == True:
					self.alpha[u] += self.learning_rate * (error - self.k_b * self.alpha[u])
					self.beta[m] += self.learning_rate * (error - self.k_b * self.beta[m])
			current_error = self.RMSE(X_val, Y_val)
			if current_error < cost:
				cost = current_error
			else:
				break
		return cost
