import csv
import numpy as np
from keras.utils import np_utils


occ_mapping = None
movies = None
users = None
base_dir = None


class Movie:
	def __init__(self, row):
		self.movie_id = int(row[0])
		self.movie_title = row[1]
		self.genres = map(lambda x: int(x), row[5:])

	def get_vector(self):
		return self.genres


class Rating:
	def __init__(self, row):
		self.userid = int(row[0])
		self.itemid = int(row[1])
		self.rating = int(row[2])
		self.timestamp = row[3]


class User:
	def __init__(self, row):
		global occ_mapping
		self.userid = int(row[0])
		limits = [14, 21, 28, 36, 48, 55, 65, 73]
		base = [0] * len(limits)
		for i, l in enumerate(limits):
			if int(row[1]) < l:
				base[i] = 1
				break
		self.age = base
		if row[2] == 'M':
			self.gender = [1,0]
		else:
			self.gender = [0,1]
		base2 = [0] * len(occ_mapping)
		base2[occ_mapping[row[3]]] = 1
		self.occupation = base2
		self.zip_code = row[4]

	def get_vector(self):
		return self.gender + self.age + self.occupation


def load_data(file_path, data_type, delim):
	rows = []
	with open(file_path, 'rb') as csvfile:
		datarows = csv.reader(csvfile, delimiter=delim)
		for row in datarows:
			if len(row):
				if data_type:
					rows.append(data_type(row))
				else:
					rows.append(row)
	return rows


def plain_read(file_path):
	rows = []
	with open(file_path, 'rb') as f:
		for row in f:
			rows.append(row.rstrip())
	mapping = {x:i for i,x in enumerate(rows)}
	return mapping


def init(bdir = './ml-100k/'):
	global occ_mapping, movies, users, base_dir
	base_dir = bdir
	occ_mapping = plain_read(base_dir + "u.occupation")
	movies = load_data(base_dir + "u.item", Movie, '|')
	users = load_data(base_dir + "u.user", User, '|')
	movies = {x.movie_id:x for x in movies}
	users = {x.userid:x for x in users}
	ratings = load_data(base_dir + "u" + str(1) + ".base", Rating, '\t')


def get_data_split(i):
	global base_dir, movies, users
	assert(i >=1 and i <= 5)
	train_ratings = load_data(base_dir + "u" + str(i) + ".base", Rating, '\t')
	test_ratings = load_data(base_dir + "u" + str(i) + ".test", Rating, '\t')
	X_train, Y_train = [], []
	X_test, Y_test = [], []
	for rating in train_ratings:
		X_train.append(users[rating.userid].get_vector() + movies[rating.itemid].get_vector())
		Y_train.append(rating.rating - 1)
	for rating in test_ratings:
		X_test.append(users[rating.userid].get_vector() + movies[rating.itemid].get_vector())
		Y_test.append(rating.rating - 1)
	Y_train = np_utils.to_categorical(Y_train, 5)
	Y_test = np_utils.to_categorical(Y_test, 5)
	return (np.array(X_train), Y_train), (np.array(X_test), Y_test)


if __name__ == "__main__":
	init()
	for i in range(1,6):
		get_data_split(i)
