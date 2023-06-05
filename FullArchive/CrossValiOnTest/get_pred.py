import json
from metrics import test
import os
def get_pred():
    folders = ["insurance_K=10_new"]
    # folders = ["banking_K=10","insurance_K=10"]

    for folder in folders:
        for epoch in range(10):
            file_name = "{}/xpredictions-{}.json".format(folder,epoch)
            with open(file_name) as fIn:
                data = json.load(fIn)["data"]
                os.makedirs("{}/prediction".format(folder,epoch),exist_ok=True)
                with open("{}/prediction/predictions-{}.json".format(folder,epoch),"w") as fOut:
                    for pred in data:
                        fOut.write(json.dumps(pred)+"\n")
                test("{}/prediction/predictions-{}.json".format(folder,epoch))


get_pred()
