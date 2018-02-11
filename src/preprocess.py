with open('/home/shikhar/Movie-Recommendation/MovieLens 26M Dataset/ratings.csv', 'r') as a, open('/home/shikhar/Movie-Recommendation/MovieLens 26M Dataset/train.csv', 'w') as b, open('/home/shikhar/Movie-Recommendation/MovieLens 26M Dataset/test.csv', 'w') as c:
	csvfile = a.readlines()
	b.write(csvfile[0])
	c.write(csvfile[0])
	i = 1
	while i < len(csvfile):
		if csvfile[i][0] == csvfile[i + 4][0]:
			b.write(csvfile[i])
			b.write(csvfile[i + 1])
			c.write(csvfile[i + 2])
			b.write(csvfile[i + 3])
			b.write(csvfile[i + 4])
			i += 4
		else:
			b.write(csvfile[i])
		i += 1