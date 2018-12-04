import numpy as np
import pickle,pdb

def return_data(mode):

	return_dict = {}
	contents = []

	with open('preprocess.pkl', 'rb') as newf:
		data = pickle.load(newf)

	for i in range(0,len(data)):
	   		
	    	return_dict = {
	    	'video':data[i][4],
	    	'audioMagnitude':data[i][5],
	    	'audioPhase':data[i][8],
	    	'cleanAudio':data[i][6]
	    	}    	

	    	# return_dict[i]['video'] = data[i][4]
	    	# return_dict[i]['mixed'] = data[i][5]
	    	# return_dict[i]['speech'] = data[i][6]
	    	contents.append(return_dict)
	pickle_out = open("extracted_new.pickle","wb")
	pickle.dump(contents, pickle_out)
	pickle_out.close()
	pdb.set_trace()
	trainSize = int(0.8*len(contents))
	contents = np.array(contents)
	if mode is 'train':	
		return contents[:trainSize,0],contents[:trainSize,1],contents[:trainSize,2],contents[:trainSize,-1]

def main():
	a,b,c,d = return_data('train')

if __name__ == '__main__':
	main()