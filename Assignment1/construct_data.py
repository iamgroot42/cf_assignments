import numpy as np


def user_item_matrix(users, ratings, movies):
	matrix = np.zeros((len(users), len(movies)))
	for rating in ratings:
		matrix[rating.userid-1][rating.itemid-1] = rating.rating
	return matrix


def item_category_matrix(items):
	matrix = []
	for movie in items:
		matrix.append([int(y) for y in movie.genres])
	matrix = np.array(matrix)
	return matrix


# def user_category(users, ratings):
# 	matrix = []
# 	for user in users:
# 		categories = [[0,0]] * len(ratings[0].genres)
# 		for rating in ratings:
# 			if rating.userid == user.userid:
# 				categories
