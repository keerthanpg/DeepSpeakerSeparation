import pickle
import network
import numpy as np

f = open('extracted_new.pickle', 'rb')
data = pickle.load(f)

data_new = []

for i in range(len(data)):
	noisy_imag = data[i]['noisyPhase'].imag
	noisy_real = data[i]['noisyPhase'].real
	data[i].pop('noisyPhase') 
	data[i]['noisyPhase'] = np.arctan(np.divide(noisy_imag,noisy_real))	
	

	clean_imag = data[i]['cleanPhase'].imag
	clean_real = data[i]['cleanPhase'].real
	data[i].pop('cleanPhase') 
	data[i]['cleanPhase'] = np.arctan(np.divide(clean_imag,clean_real))

	sample = []
	sample.append(data[i]['video'])
	sample.append(data[i]['noisyMagnitude'])
	sample.append(data[i]['cleanMagnitude'])
	sample.append(data[i]['noisyPhase'])
	sample.append(data[i]['cleanPhase'])
	data_new.append(sample)



pickle.dump(data_new, open('processed.pickle', 'wb'))