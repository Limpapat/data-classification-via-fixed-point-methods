import json

setting = {
    "data" : [
        {
            "name" : "sample_generated_data",
            "path" : "source/sample_generated_data.csv",
            "class_label" : ["class_1", "class_2", "class_3"]
        },
        {
            "name" : "iris",
            "path" : "source/iris.data",
        },
        {
            "name" : "abalone",
            "path" : "source/abalone.data",
            "target" : 0
        },
        {
            "name" : "parkinsons",
            "path" : "source/parkinsons.data",
            "target" : 17
        },
        {
            "name" : "heart_disease",
            "path" : "source/heart.csv",
        },
        ],
    "model" : "MLP",
    "loss" : "MCE",
    "penalty" : "l1",
    "n_test" : .1,
    "n_iter" : 100,
    "disp" : False,
    "optimizers" : {
        "SGD" : {
            "lr" : 2.,
            "rp" : 1e-3,
            "cseq" : None
        },
        "FBA" : {
            "lr" : 2.,
            "rp" : 1e-3,
            "cseq" : './cseq.json'
        },
        "SFBA" : {
            "lr" : 2.,
            "rp" : 1e-3,
            "cseq" : './cseq.json'
        },
        "ISFBA" : {
            "lr" : 2.,
            "rp" : 1e-3,
            "cseq" : './cseq.json'
        },
        "PFBA" : {
            "lr" : 2.,
            "rp" : 1e-3,
            "cseq" : './cseq.json'
        },
        "IPFBA" : {
            "lr" : 2.,
            "rp" : 1e-3,
            "cseq" : './cseq.json'
        },
        "ParallelSFBA" : {
            "lr" : 2.,
            "rp" : 1e-3,
            "cseq" : './cseq.json'
        },
        "ParallelISFBA" : {
            "lr" : 2.,
            "rp" : 1e-3,
            "cseq" : './cseq.json'
        },
    }
}

with open('setting.json', 'w') as f:
    f.write(json.dumps(setting))
    f.close()