import pandas as pd
from pre import generate_cols, prep_df
import tqdm
# from IPython import embed
real_data_path = 'Online_Retail.csv'
df = pd.read_csv(real_data_path, names=list(range(0,6)))
data = df.values
test_data = data[:1000]
test_df = df.iloc[:1000]

def generateTwoItemsets(oneItemsets, miniSupport, itemFreq):
    ## if 1-itemset is not frequent(< miniSupport), skip it
    num = len(oneItemsets)
    twoItemsets = []
    for i in range(num-1):
        item1 = oneItemsets[i]
        if itemFreq[item1] < miniSupport:
            continue
        for j in range(i+1, num):
            item2 = oneItemsets[j]
            if itemFreq[item2] < miniSupport:
                continue
            twoItemsets.append((item1, item2))
    return twoItemsets

def oneItemsetFreq(data):
    #count item occurance
    dataFlat = data.flatten()
    import collections as cl
    itemFreq = cl.Counter(dataFlat)
    return itemFreq

def twoItemsetFreq(df, itemFreq, miniSupport):
    total, _ = df.shape
    twoItemFreq = {}
    for i in range(total):
        items = list(df.iloc[i])
        pairs = generateTwoItemsets(items, miniSupport, itemFreq)
        for pair in pairs:
            item1, item2 = pair
            twoItemFreq[(item1, item2)] = twoItemFreq.get((item1, item2), 0) + 1
    return twoItemFreq
    

def computeConfidence(itemFreq, twoItemsets, item1, item2):
    confidence1 = twoItemsets[(item1, item2)] / itemFreq[item1]
    confidence2 = twoItemsets[(item1, item2)] / itemFreq[item2]
    return confidence1, confidence2
    
total, _ = df.shape    
miniSupport = 50
oneItemFreq = oneItemsetFreq(data)
# oneItemFreq = oneItemsetFreq(test_data)
twoItemsets = twoItemsetFreq(df, oneItemFreq, miniSupport)
item_a = []
item_b = []
support_ab = []
confidence_a_to_b = []
confidence_b_to_a = []
for pair, support in twoItemsets.items():
    itema, itemb = pair
    item_a.append(itema)
    item_b.append(itemb)
    support_ab.append(support/ total * 1.0)
    conf1, conf2 = computeConfidence(oneItemFreq, twoItemsets, itema, itemb)
    confidence_a_to_b.append(conf1)
    confidence_b_to_a.append(conf2)

cols = generate_cols()
outdata = [pd.DataFrame(item_a), pd.DataFrame(item_b),pd.DataFrame(support_ab), pd.DataFrame(confidence_a_to_b),pd.DataFrame(confidence_b_to_a) ]
output =  prep_df(outdata, cols)
print(output.head())
