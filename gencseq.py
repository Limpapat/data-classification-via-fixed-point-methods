import json
cseq = {
    "alpha" : ".5",
    "beta" : ".5",
    "theta" : "sqrt(1+4*n*n)/(1+sqrt(1+4*(n+1)*(n+1)))"
}

# setting = {
#     "data" : "source/sample_generated_data.csv",
#     "model" : "MLP",
#     "loss" : "MCE",
#     "penalty" : "l1",
#     "n_test" : .1,
#     "n_iter" : 100,
#     "optimizers" : {
#         "SGD" : {
#             "lr" : 2.,
#             "rp" : 1e-3,
#             "cseq" : None
#         },
#         "FBA" : {
#             "lr" : 2.,
#             "rp" : 1e-3,
#             "cseq" : './cseq.json'
#         },
#         "SFBA" : {
#             "lr" : 2.,
#             "rp" : 1e-3,
#             "cseq" : './cseq.json'
#         },
#         "ISFBA" : {
#             "lr" : 2.,
#             "rp" : 1e-3,
#             "cseq" : './cseq.json'
#         },
#         "PFBA" : {
#             "lr" : 2.,
#             "rp" : 1e-3,
#             "cseq" : './cseq.json'
#         },
#         "IPFBA" : {
#             "lr" : 2.,
#             "rp" : 1e-3,
#             "cseq" : './cseq.json'
#         },
#         "ParallelSFBA" : {
#             "lr" : 2.,
#             "rp" : 1e-3,
#             "cseq" : './cseq.json'
#         },
#         "ParallelISFBA" : {
#             "lr" : 2.,
#             "rp" : 1e-3,
#             "cseq" : './cseq.json'
#         },
#     }
# }

with open('cseq.json', 'w') as f:
    f.write(json.dumps(cseq))
    f.close()