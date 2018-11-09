import pandas as pd 
import numpy as np
from IPython import embed

class PreProcessing(object):
    def __init__(self, path):
        data = pd.read_csv(path)
        del data['Id']
        colums = data.columns
        m,n = data.shape
        embed()
        



if __name__ == '__main__':
    dataProcessing = PreProcessing('/Users/hy/Downloads/all/train_V2.csv')
