import argparse, json
import datetime
import os
import logging
import torch, random
import matplotlib.pyplot as plt

from server import *
from client import *
import models, datasets

	

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()

	los = []
	epoch = []
	accuracy = []

	with open(args.conf, 'r') as f:
		conf = json.load(f)	
	
	
	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
	
	server = Server(conf, eval_datasets)
	clients = []
	
	for c in range(conf["no_models"]):
		clients.append(Client(conf, server.global_model, train_datasets, c))
		
	print("\n\n")
	for e in range(conf["global_epochs"]):
		epoch.append(e)
		candidates = random.sample(clients, conf["k"])
		
		weight_accumulator = {}
		
		for name, params in server.global_model.state_dict().items():
			weight_accumulator[name] = torch.zeros_like(params)
		
		for c in candidates:
			diff = c.local_train(server.global_model)
			
			for name, params in server.global_model.state_dict().items():
				weight_accumulator[name].add_(diff[name])
				
		
		server.model_aggregate(weight_accumulator)
		
		acc, loss = server.model_eval()
		accuracy.append(acc)
		los.append(loss)
		print("Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

	print(f'\nPlotting Accuracy vs Epochs')
	plt.plot(epoch,accuracy, c='r')

	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	#plt.legend()
	plt.show()

	plt.plot(epoch,los, c='b')

	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	# plt.legend()
	plt.show()

#python main.py -c ./utils/conf.json