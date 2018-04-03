#!/usr/bin/python3


print("--------------------------Pre-Processing Data--------------------------")

try:

	with open('../MovieLens 10M Dataset/ratings.csv', 'r') as y, open('../MovieLens 10M Dataset/ratings_cleaned.csv', 'w') as z:
		csvfile = y.readlines()
		z.write(csvfile[0])
		i = 1
		movieid = {}
		counter = 1
		while i < len(csvfile):
			lister = csvfile[i].split(',')
			try:
				lister[1] = movieid[lister[1]]
			except:
				movieid[lister[1]] = str(counter)
				lister[1] = str(counter)
				counter += 1
			csvfile[i] = ','.join(lister)
			z.write(csvfile[i])
			i += 1
	y.close()
	z.close()

except Exception as ex:

	print(ex)
	print("----------------------Failed To Normalize MovieId----------------------")
	exit()

else:	

	print("--------------------MovieId Normalized Successfully--------------------")


try:

	with open('../MovieLens 10M Dataset/ratings_cleaned.csv', 'r') as a, open('../MovieLens 10M Dataset/train.csv', 'w') as b, open('../MovieLens 10M Dataset/test.csv', 'w') as c:
		csvfile = a.readlines()
		b.write(csvfile[0])
		c.write(csvfile[0])
		i = 1
		while i < len(csvfile):
			if i + 4 < len(csvfile) and csvfile[i].split(',')[0] == csvfile[i + 4].split(',')[0]:
				b.write(csvfile[i])
				b.write(csvfile[i + 1])
				c.write(csvfile[i + 2])
				b.write(csvfile[i + 3])
				b.write(csvfile[i + 4])
				i += 4
			else:
				b.write(csvfile[i])
			i += 1
	a.close()
	b.close()
	c.close()

except Exception as ex:

	print(ex)
	print("-----------------------Failed To Prepare Dataset-----------------------")
	exit()

else:	

	print("---------------Train and Test Data Prepared Successfully---------------")
