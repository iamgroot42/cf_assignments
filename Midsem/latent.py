import numpy as np
from tqdm import tqdm
import parser


def nmae(y, y_, minr=1, maxr=5):
	return np.sum(np.absolute(y-y_))/(len(y) * (maxr-minr))


class LatentFactorization:
	def __init__(self, Y, n_latent=25, lambda_u=0.1, lambda_v=0.1,alpha_scale=1.01, max_iters=10):
		self.n_latent = n_latent
		self.lambda_u = lambda_u
		self.lambda_v = lambda_v
		self.Y = Y
		self.R = self.Y != 0
		self.alpha_scale = alpha_scale
		self.max_iters = max_iters
		self.U = np.random.normal(size=(self.Y.shape[0],self.n_latent))
		self.V = np.random.normal(size=(self.n_latent,self.Y.shape[1]))
		self.u_g = np.mean(self.Y)
		self.b_n = []
		self.b_m = []
		# Calculate biases
		for i in range(self.Y.shape[0]):
			self.b_m.append(np.mean(Y[i,:]) - self.u_g)
		for i in range(self.Y.shape[1]):
			self.b_n.append(np.mean(Y[:,i]) - self.u_g)
		# Normalize data
		for i in range(self.Y.shape[0]):
			for j in range(self.Y.shape[1]):
				if Y[i][j]:
					self.Y[i][j] -= self.b_n[j] + self.b_m[i] + self.u_g

	# Solver xA=B for x, return x
	def solver(self, A, B):
		solution = np.linalg.lstsq(A.T, B.T)[0].T
		return solution

	def soft(self, t, s):
		return np.sign(t) * np.maximum(np.maximum(t, 0),-s)

	def error(self):
		error = np.power(np.linalg.norm(self.Y - self.R * np.dot(self.U, self.V)),2)
		error += self.lambda_u * np.power(np.linalg.norm(self.U),2)
		error += self.lambda_v * np.linalg.norm(self.V, 1)
		return error

	def solve(self):
		current_error = np.inf
		for _ in range(self.max_iters):
			# print(current_error)
			beta = 1.0
			UV = np.dot(self.U, self.V)
			Z = UV + (1/beta) * (self.R * (self.Y - UV))
			self.U = (self.solver(np.dot(Z, self.V.T), np.dot(self.V, self.V.T) + self.lambda_u * np.eye(self.V.shape[0]))).T
			UV = np.dot(self.U, self.V)
			W = UV + (1/beta) * (self.R * (self.Y - UV))
			alpha = np.amax(np.linalg.eig(np.dot(self.U.T,self.U))[0]) * self.alpha_scale
			self.V = self.soft(self.V + (1/alpha) * np.dot(self.U.T, (W - np.dot(self.U, self.V))), self.lambda_v/(2 * alpha))
			error = self.error()
			current_error = error

	#Get prediction for user i on item j
	def get_rating(self, i, j):
		# return np.inner(self.U[j], self.V[:,i])
		return self.u_g + self.b_n[i] + self.b_m[j] + np.inner(self.U[j], self.V[:,i])

	def predict(self, X):
		y = []
		y_ = []
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				if X[i][j] != 0:
					y.append(X[i][j])
					y_.append(np.clip(self.get_rating(i,j), 1, 5))
		return np.array(y), np.array(y_)


if __name__ == "__main__":
	import sys
	dd = sys.argv[1]
	movies = parser.load_data(dd + "u.item", parser.Movie, '|')
	ratings = []
	for i in range(1,1+5):
		ratings.append( [parser.load_data(dd + "u" + str(i) + ".base", parser.Rating, '\t'), parser.load_data(dd + "u" + str(i) + ".test", parser.Rating, '\t')] )
	users = parser.load_data(dd + "u.user", parser.User, '|')
	latent_grid = [5]
	lambda_values = [1,1e-1,1e-2,1e-3,1e-4]
	for latent in latent_grid:
		for lambda_val in lambda_values: 
			errors = []
			for fold in ratings:
				#Transpose ratings to convert standard problem algorithm 
				R = parser.R_matrix(len(users), len(movies), fold[0]).T
				R_test = parser.R_matrix(len(users), len(movies), fold[1])
				test = fold[1]
				lf = LatentFactorization(R, max_iters=100, n_latent=latent,lambda_u=lambda_val,lambda_v=lambda_val)
				lf.solve()
				y, y_ = lf.predict(R_test)
				errors.append(nmae(y, y_))
				# print("NMAE:",errors[-1])
			print("For latent",latent,"and lambda",lambda_val,"average NMAE over all folds:", sum(errors)/len(errors)*1.0)
