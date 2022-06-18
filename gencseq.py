import json
cseq = {
    "alpha" : "sqrt(1+4*n*n)/(1+sqrt(1+4*(n+1)*(n+1)))",
    "theta" : "0.5**n"
}

with open('cseq.json', 'w') as f:
    f.write(json.dumps(cseq))
    f.close()