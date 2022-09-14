# FedSelection
Uses command
```
python main.py -c ./utils/conf.json
```
to run the script

Only requires **Torch** and **Matplot**

Parameters can be modified in JSON file under utils folder:
```buildoutcfg
"model_name" : model that is being used for local client, full list in models.py

"no_models" : How many simulated clients that will participate in federated training (Recommend less than 100 for devices with less than 12GB graphic card memory)
	
"type" : Dataset name that will be used 
	
"global_epochs" : Global training round
	
"local_epochs" : Local training round
	
"k" : how many clients will be selected each global training round
	
"batch_size" : local batch size
	
"lr" : learning rate
	
"momentum" : Local training parameter
	
"lambda" : Local training parameter

"coeff" : dynamic weight modifier

```