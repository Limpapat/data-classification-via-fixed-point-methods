import json
cseq = {
    "alpha" : ".5",
    "beta" : ".5",
    "theta" : "sqrt(1+4*n*n)/(1+sqrt(1+4*(n+1)*(n+1)))"
}

with open('cseq.json', 'w') as f:
    f.write(json.dumps(cseq))
    f.close()