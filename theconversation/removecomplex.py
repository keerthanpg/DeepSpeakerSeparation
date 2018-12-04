import pickle
import network

f = open('extracted_new.pickle', 'rb')
data = pickle.load(f)

data_new = []

for i in range(len(data)):
	data[i]['noisyPhaseImag'] = data[i]['noisyPhase'].imag
	data[i]['noisyPhaseReal'] = data[i]['noisyPhase'].real
	data[i].pop('noisyPhase') 
	
	data[i]['cleanPhaseImag'] = data[i]['cleanPhase'].imag
	data[i]['cleanPhaseReal'] = data[i]['cleanPhase'].real
	data[i].pop('cleanPhase')
	sample = []
	sample.append(data[i]['video'])
	sample.append(data[i]['noisyMagnitude'])
	sample.append(data[i]['cleanMagnitude'])
	sample.append(data[i]['noisyPhaseImag'])
	sample.append(data[i]['noisyPhaseReal'])
	sample.append(data[i]['cleanPhaseImag'])
	sample.append(data[i]['cleanPhaseReal'])

pickle.dump(data, open('processed.pickle', 'wb'))