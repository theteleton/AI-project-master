import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec
import string
import dgl
import torch
from parameters import args


nltk.download('punkt')
nltk.download('stopwords')

def preprocess_sentence(sentence):
    words = word_tokenize(sentence)
    words = [word.lower() for word in words]
    words = [word for word in words if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return words

def preprocess(path):
    df = pd.read_csv(path)
    documents = []
    preprocessed_sentences = []
    labels = []
    embedds = []

    for i in range(len(df)):
        text = df.iloc[i, 2]
        label = df.iloc[i, 0]
        labels.append(label)
        sentences = sent_tokenize(text)
        document_preprocess = []
        for sentence in sentences:
            preprocessed_sentences.append(preprocess_sentence(sentence))
            document_preprocess.append(preprocess_sentence(sentence))
    
        documents.append(document_preprocess)
    model = Word2Vec(sentences=preprocessed_sentences, vector_size=args.in_dim, window=args.window, min_count=1, workers=4)
    edges_w2w = [[], []]
    edges_w2d = [[], []]
    dict = {}
    words = 0
    print("START")
    for i, doc in enumerate(documents):
        if i % 100 == 0:
            print(i)
        j = 0

        for sent in doc:
            for k, word in enumerate(sent):
                if word not in list(dict.keys()):
                    dict[word] = words
                    embedds.append(model.wv[word])
                    words += 1
                    
                edges_w2d[0].append(i)
                edges_w2d[1].append(dict[word])
                for nword in sent[k + 1:k+args.window]:
                    if nword not in list(dict.keys()):
                        dict[nword] = words
                        embedds.append(model.wv[nword])
                        words += 1
                    edges_w2w[0].append(dict[word])
                    edges_w2w[1].append(dict[nword])
    
    embednp = np.array(embedds)
    w2wedges = np.array(edges_w2w)
    w2dedges = np.array(edges_w2d)
    labelsnp = np.array(labels)

    np.savetxt(embednp, "./data/embeddings.txt")
    np.savetxt(w2wedges, "./data/w2w.txt")
    np.savetxt(w2dedges, "./data/w2d.txt")
    np.savetxt(labelsnp, "./data/labels.txt")

    docs = int(len(df))
    print(words)
    print(docs)
    

                    

                