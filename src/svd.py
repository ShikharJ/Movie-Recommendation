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

	def predict(self, u, i):
		x = numpy.dot(self.U[u], self.M[i])
		if self.bias == True:
			x += self.mu + self.alpha[u] + self.beta[i]
		return x

	def train(self, X, Y):
		Basic.preprocess(self, X, Y)
		self.U = numpy.empty((self.u_len + 1, self.f))
		self.M = numpy.empty((self.m_len + 1, self.f))
		self.alpha = numpy.zeros(self.u_len + 1)
		self.beta = numpy.zeros(self.m_len + 1)
		initial_val = numpy.random.uniform(-0.01, 0.01)
		for row in self.U:
			row.fill(initial_val)
		for row in self.M:
			row.fill(initial_val)
		return None
