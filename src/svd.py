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

	def predict(self, u, i):
		x = numpy.dot(self.U[u], self.M[i])
		if self.bias == True:
			x += self.mu + self.alpha[u] + self.beta[i]
		return x

	def test_train_split(self, X, n):
		numpy.random.shuffle(X)
		l = (int)(X.shape[0] * (n - 1) / n)
		self.X_train, self.X_test = X[:l, :], X[l:, :]
		self.Y_test = self.X_test[:, 2]

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
		self.test_train_split(X, 5)
		cost = float('inf')
		U_temp = numpy.zeros(self.dim)
		for r in range(self.epochs):
			for x in self.X_train:
				u, m = x[0], x[1]
				error = R[u][m] - self.predict(u, m)
				U_temp = self.U[u] + self.learning_rate * (error * self.M[m] - self.U[u] * self.k_u)
				self.M[m] += self.learning_rate * (error * self.U[u] - self.M[m] * self.k_m)
				self.U[u] = U_temp[:]
				if self.bias == True:
					self.alpha[u] += self.learning_rate * (error - self.k_b * self.alpha[u])
					self.beta[m] += self.learning_rate * (error - self.k_b * self.beta[m])
			current_error = self.validation_error()
			if current_error < cost:
				cost = current_error
			else:
				break

	def validation_error(self):
		return self.RMSE(self.X_test, self.Y_test)
