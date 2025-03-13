from utils.synhtesizer.ttvae import TTVAE
from sklearn.datasets import load_iris

dataset = load_iris(as_frame=True).data

ttvae_model = TTVAE()
ttvae_model.fit(dataset)
ttvae_model.save('save.pt')

ttvae_loaded = TTVAE.load('save.pt')

synthetic_data = ttvae_loaded.sample(1000)
synthetic_data.to_csv('data_synth.csv', index=False)