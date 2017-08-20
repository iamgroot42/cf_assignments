import os
import csv
import sys

import construct_data
import algos

class Movie:
	def __init__(self, row):
		self.movie_id = int(row[0])
		self.movie_title = row[1]
		self.release_data = row[2]
		self.video_release_data = row[3]
		self.imdb_url = row[4]
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


def cross_vals(users, ratings, movies, algo):
	errors = []
	partials = [construct_data.user_item_matrix(users, ratings[i][0], movies) for i in range(5)]
	print partials[0]
	for i in range(5):
		errors.append(algo(ratings[i][1][:50], partials[i]))
	return errors, sum(errors) / 5.0


if __name__ == "__main__":
	dd = sys.argv[1]
	movies = load_data(dd + "u.item", Movie, '|')
	ratings = []
	for i in range(1,1+5):
		ratings.append( [load_data(dd + "u" + str(i) + ".base", Rating, '\t'), load_data(dd + "u" + str(i) + ".test", Rating, '\t')] )
	users = load_data(dd + "u.user", User, '|')
	genres = load_data(dd + "u.genre", None, '|')
	ic = construct_data.item_category_matrix(movies)
	print cross_vals(users, ratings, movies, algos.user_user)
