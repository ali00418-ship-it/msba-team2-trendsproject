import pandas as pd

def clip(filename):

    df = pd.read_csv(filename, nrows=1000)

    df.to_csv('clipped.csv')

clip('complaints.csv')