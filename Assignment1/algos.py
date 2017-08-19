import numpy as np


def mae(y, y_):
	return np.absolute(y-y_)/len(y)

def pearson(thing_1, thing_2):
	indices = np.nonzero(thing_1*thing_2!=0)
	thing1 = thing_1[indices]
	thing2 = thing_2[indices]
	mean1 = np.mean(thing1)
	mean2 = np.mean(thing2)
	numerator = np.sum((thing1-mean1)-(thing2-mean2))
	denominator = np.sqrt(np.sum(np.power(thing1-mean1),2) * np.sum(np.power(thing2-mean2),2))
	return numerator / denominator


def user_user(predict, ground_truth):
	errors=[]
	for rating in predict:
		active_ratings = ground_truth[rating.userid-1]
		r_ua = np.mean(active_ratings[np.nonzero(active_ratings)])
		relevant = np.nonzero(ground_truth[:,rating.itemid-1])
		k_ah = np.array([pearson(ground_truth[rating.userid-1,:],ground_truth[i,:]) for i in relevant])
		r_uh = np.array([ (ground_truth[i][rating.itemid-1] - np.mean(ground_truth[i][np.nonzero(ground_truth[i])])) for i in relevant])
		CF = r_ua + np.mean(np.multiply(k_ah,r_ua)) / np.mean(np.absolute(k_ah))
		difference = rating.rating - np.round(CF)
		errors.append(difference)
