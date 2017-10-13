import os
import csv
import sys

import numpy as np
from sets import Set
import random

changed_movieids = None

def load_data(file_path, data_type, delim):
	rows = []
	with open(file_path, 'rb') as file:
		for rawrow in file:
			row = rawrow.rstrip().split(delim)
			if len(row):
				if data_type:
					rows.append(data_type(row))
				else:
					rows.append(row)
	return rows


def data_split_save(onem, hk):
	global changed_movieids
	rows = load_data(onem + 'ratings.dat', None, "::")
	random.shuffle(rows)
	for row in rows:
		row[1] = changed_movieids[row[1]]
	size = len(rows) / 5
	for i in range(5):
		test = rows[i * size: (i+1)*size]
		train = rows[:i*size] + rows[(i+1)*size:]
		with open(hk + "u" + str(i+1) + ".base", 'w') as file:
			writer = csv.writer(file, delimiter='\t')
			writer.writerows(train)
		with open(hk + "u" + str(i+1) + ".test", 'w') as file:
			writer = csv.writer(file, delimiter='\t')
			writer.writerows(test)


def user_save(onem, hk):
	rows = load_data(onem + 'users.dat', None, "::")
	with open(hk + "u.user", 'w') as file:
		writer = csv.writer(file, delimiter='|')
		for row in rows:
			writer.writerow([row[0], row[2]])


def get_category_boolmap(catstring, categories):
	boolarr = [0] * len(categories)
	catstring = catstring.split("|")
	for cat in catstring:
		boolarr[categories[cat]] = 1
	return boolarr


def movies_genres_save(onem, hk):
	global changed_movieids
	rows = load_data(onem + 'movies.dat', None, "::")
	movieids = [x[0] for x in rows]
	changed_movieids = {movieids[i]:i for i in range(len(movieids))}
	categories = []
	for row in rows:
		categories += row[2].split("|")
	categories = list(Set(categories))
	categories = {categories[i]:i for i in range(len(categories))}
	with open(hk + "u.genre", 'w') as file:
		writer = csv.writer(file, delimiter='|')
		for cat in categories.keys():
			writer.writerow([cat, categories[cat]])	
	with open(hk + "u.item", 'w') as file:
		writer = csv.writer(file, delimiter='|')
		for row in rows:
			writer.writerow([changed_movieids[row[0]], row[1], " ", " ", " "] + get_category_boolmap(row[2], categories))


if __name__ == "__main__":
	movies_genres_save(sys.argv[1], sys.argv[2])
	data_split_save(sys.argv[1], sys.argv[2])
	user_save(sys.argv[1], sys.argv[2])
