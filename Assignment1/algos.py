import numpy as np
from tqdm import tqdm


def nmae(y, y_, minr=1, maxr=5):
	return np.sum(np.absolute(y-y_))/(len(y) * (maxr-minr))


def pearson(thing_1, thing_2):
	indices = np.nonzero(thing_1*thing_2!=0)
	thing1 = thing_1[indices]
	thing2 = thing_2[indices]
	mean1 = np.mean(thing1)
	mean2 = np.mean(thing2)
	numerator = np.sum((thing1-mean1)-(thing2-mean2))
	denominator = np.sqrt(np.sum(np.power((thing1-mean1),2)) * np.sum(np.power((thing2-mean2),2)))
	if numerator == 0 and denominator == 0:
		return 0
	return numerator / denominator


def user_user(predict, ground_truth, minr=1, maxr=5):
	p = []
	gt = []
	for rating in tqdm(predict):
		active_ratings = ground_truth[rating.userid-1,:]
		r_ua = np.mean(active_ratings[np.nonzero(active_ratings)])
		relevant = np.nonzero(ground_truth[:,rating.itemid-1])[0]
		k_ah = np.array([pearson(ground_truth[rating.userid-1,:],ground_truth[i,:]) for i in relevant])
		r_uh = np.array([ (ground_truth[i][rating.itemid-1] - np.mean(ground_truth[i][np.nonzero(ground_truth[i])])) for i in relevant])
		CF = r_ua + np.mean(np.multiply(k_ah,r_uh)) / np.mean(np.absolute(k_ah))
		gt.append(rating.rating)
		p.append(max(minr,min(maxr,np.round(CF))))
	return nmae(np.array(gt), np.array(p))


def item_item(predict, ground_truth, minr=1, maxr=5):
	p = []
	gt = []
	for rating in tqdm(predict):
		temp = ground_truth[rating.userid-1,:]
		r_ua_bar = np.mean(temp[np.nonzero(temp)])
		active_ratings = ground_truth[:,rating.itemid-1]
		r_ia = np.mean(active_ratings[np.nonzero(active_ratings)])
		relevant = np.nonzero(ground_truth[rating.userid-1,:])[0]
		u_ah = np.array([pearson(ground_truth[:,rating.itemid-1],ground_truth[:,i]) for i in relevant])
		r_ua = np.array([ (ground_truth[rating.userid-1][i] - r_ua_bar) for i in relevant])
		CF = r_ia + np.mean(np.multiply(u_ah,r_ua)) / np.mean(np.absolute(u_ah))
		gt.append(rating.rating)
		p.append(max(minr,min(maxr,np.round(CF))))
	return nmae(np.array(gt), np.array(p))