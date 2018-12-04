import pickle

with open('extracted.pickle', 'rb') as f:
	print(type(f))
	data = pickle.load(f)

for i in range(len(data)):
	print(type(data[i]['mixed']))
	for j in range(len(data[i]['mixed'])):
		print(j)

	print(type(data[i]['speech']))
	for j in range(len(data[i]['speech'])):
		print(j)

	print(type(data[i]['mixed'][0]))




	
	break