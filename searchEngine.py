import spacy
import pandas as pd
from tqdm import tqdm
import csv
pd.set_option('display.max_colwidth', None)
from rank_bm25 import BM25Okapi
import numpy as np
from IPython.core.display import display, HTML
import time
import json
import sys
import pickle



if __name__ == "__main__":
    query = sys.argv[1]
    #print(query)
    nlp = spacy.load("fr_core_news_lg")
    df = pd.read_csv('lessurligneurs.csv')
    
    with open("toktext", 'rb') as f:
        tok_text = pickle.load(f)
      
    bm25 = BM25Okapi(tok_text)

    #query = "Convention France macron"
    tokenized_query = query.lower().split(" ")


    t0 = time.time()
    results = bm25.get_top_n(tokenized_query, df.NER.values, n=10)
    t1 = time.time()
    print(f'Searched 767 records in {round(t1-t0,3) } seconds \n')
    for i in results:
        index = np.where(df.NER.values==i)
        idItem = df.idItem.values[index]
        title = df.title.values[index]
        guid = df.guid.values[index]
        print("Surlignage id is: ", idItem)
        print(guid[0])
        print(title[0])
        #createstring = "<a href=" + guid[0] + ">" + title[0] + "</a>"
        #display(HTML(createstring))
        print('\n------------------------------------------------------------------------------------------------------\n------------------------------------------------------------------------------------------------------')
