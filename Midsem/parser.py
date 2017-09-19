import csv
import numpy as np


class Movie:
	def __init__(self, row):
		self.movie_id = int(row[0])
		self.movie_title = row[1]
		self.genres = row[5:]


class Rating:
	def __init__(self, row):
		self.userid = int(row[0])
		self.itemid = int(row[1])
		self.rating = float(row[2])
		self.timestamp = row[3]


class User:
	def __init__(self, row):
		self.userid = int(row[0])
		self.age = row[1]


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


def R_matrix(U, V, data):
	R = np.zeros((U, V))
	for datum in data:
		R[datum.userid-1][datum.itemid-1] = datum.rating
	return R
