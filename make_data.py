import numpy as np
import pandas as pd

def make_data(N, is_confused = True, confused_bin = 50):
	np.random.seed(1)

	feature = np.random.randn(N, 2)
	df = pd.DataFrame(feature, columns = ['x1', 'x2'])

	df['y'] = df.apply(lambda row : 1 if (5*row.x1 + 3*row.x2 - 1)>0 else 0,  axis=1)

	if is_confused:
		def get_model_confused(data):
			y = 1 if (data.name % confused_bin) == 0 else data.y
			return y
		df['y'] = df.apply(get_model_confused, axis=1)

	return df