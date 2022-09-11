import argparse, json
import datetime
import operator
import os
import logging
import torch, random
import matplotlib.pyplot as plt
import warnings

from server import *
from client import *
import models, datasets

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    # initiate Variables/DS
    los = []
    epoch = []
    accuracy = []
    static_weight = {}
    initial_weight = 2
    dynamic_weight = {}
    with open(args.conf, 'r') as f:
        conf = json.load(f)

    # initiate default static and dynamic weight for number of clients
    for i in range(conf["no_models"]):
        static_weight[i] = 2

    for i in range(conf["no_models"]):
        dynamic_weight[i] = static_weight[i]

    print(dynamic_weight)

    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

    server = Server(conf, eval_datasets)
    clients = []
    clients_id = []
    client_loss = {}
    sorted_client_loss = list(sorted(dynamic_weight.items(), key=operator.itemgetter(1),reverse=True))
    for c in range(conf["no_models"]):
        clients.append(Client(conf, server.global_model, train_datasets, c))



    print("\n\n")
    for e in range(conf["global_epochs"]):
        clients_id = []
        epoch.append(e)
        candidates = []

        # Original Random Sampling Method
        #candidates = random.sample(clients, conf["k"])

        #Sample based on top 10 highest loss
        sorted_client_loss = list(sorted(dynamic_weight.items(), key=operator.itemgetter(1),reverse=True))
        print("Sorted client loss this round: ", sorted_client_loss)
        for client in sorted_client_loss[:10]:
            candidates.append(clients[client[0]])

        for can in candidates:
            clients_id.append(can.client_id)
        print("Candidates selected this round:", clients_id)
        weight_accumulator = {}

        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        for c in candidates:
            diff, c_loss = c.local_train(server.global_model)
            client_loss[c.client_id] = c_loss
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])

        server.model_aggregate(weight_accumulator)
        # Return sorted client loss for each global round
        sorted_client_loss_per_global_epoch = sorted(client_loss.items(), key=lambda x: x[1], reverse=True)
        print("Sorted client loss per global epoch is: ",sorted_client_loss_per_global_epoch)
        for tup in sorted_client_loss_per_global_epoch:
            dynamic_weight[tup[0]] = tup[1]

        #sort the updated dynamic_weight dict
        #dynamic_weight = dict(sorted(dynamic_weight.items(), key=lambda item: item[1]))

        acc, loss = server.model_eval()
        accuracy.append(acc)
        los.append(loss)
        #print(dynamic_weight)
        print("Global Epoch %d, acc: %f, loss: %f\n" % (e, acc, loss))

    print(f'\nPlotting Accuracy vs Epochs')
    plt.plot(epoch, accuracy, c='r')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    # plt.legend()
    plt.show()

    plt.plot(epoch, los, c='b')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.legend()
    plt.show()

# python main.py -c ./utils/conf.json
