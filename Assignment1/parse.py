import os
import csv
import sys

import construct_data

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
		self.rating = flooat(row[2])
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
				print row
	return rows


if __name__ == "__main__":
	movies = load_data("u.item", Movie, '|')
	ratings = []
	for i in range(1,1+5):
		ratings.append([tload_data("u" + i + ".base", Rating, '\t'),[tload_data("u" + i " .text", Rating, '\t')]])
	users = load_data("u.user", User, '|')
	genres = load_data("u.genre", None, '|')
	print construct_data.user_item_matrix(users, ratings, movies)
	print construct_data.item_category_matrix(movies)
