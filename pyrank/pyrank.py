import numpy as np


def ndcg(r):
    """
    input:
        - r (list): relevance scores in rank order
    """
    # actual dcg
    adcg = dcg(r)
    # find ideal dcg
    # sort the list
    ir = sorted(r, reverse=True)
    idcg = dcg(ir)
    ndcg = adcg/idcg
    return ndcg

def dcg(r, n=None, exp=True):
    """
    input:
        - r (array): list of relevance scores in rank order
        - n (integer): DCG@n
        - exp (boolean): use exponential formula
    """
    r = np.array(r)
    if n:
        r = r[:n]
    # exponential formula?
    if exp:
        gain = 2**r +1
    else:
        gain = r
    # sum the relevance scores, logarithmically discounted by rank
    discount = np.log2(np.arange(len(r))+2)
    return np.sum(gain / discount)




if __name__=='__main__':
    r = [1.0, 1.2, 0.8, 0.4, 0.9]
    print(dcg(r))
    print(ndcg(r))
