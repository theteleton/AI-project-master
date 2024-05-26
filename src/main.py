import torch
import dgl
from parameters import args
import numpy as np 
from models import Model
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec


nltk.download('punkt')
nltk.download('stopwords')
def build_graph():
    device = torch.device("cuda:0")
    
    labels = np.loadtxt("./models/labels.txt")
    w2d = np.loadtxt("./models/w2d.txt")
    w2w = np.loadtxt("./models/w2w.txt")



    dict_graph = {
        ("word", "word2word", "word") : (torch.tensor(w2w[0].tolist(), device=device, dtype=torch.int32), torch.tensor(w2w[1].tolist(), device=device, dtype=torch.int32)),
        ("document", "word2document", "word") : (torch.tensor(w2d[0].tolist(), device=device, dtype=torch.int32), torch.tensor(w2d[1].tolist(), device=device, dtype=torch.int32)),
        ("word", "word2wordr", "word") : (torch.tensor(w2w[1].tolist(), device=device, dtype=torch.int32), torch.tensor(w2w[0].tolist(), device=device, dtype=torch.int32)),
        ("word", "word2documentr", "document") : (torch.tensor(w2d[1].tolist(), device=device, dtype=torch.int32), torch.tensor(w2d[0].tolist(), device=device, dtype=torch.int32))
    }

    return torch.tensor(labels[:15000].tolist(), device=device, dtype=torch.int32), dgl.heterograph(dict_graph).to(device)

def train(graph, labels):
    device = torch.device("cuda:0")
 
    model = Model(args.in_dim, args.h_dim, args.o_dim, args.gnn, args.heads, args.dropout, device)
    criteria = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    embeddings = np.loadtxt("./models/embeddings.txt")
    x_word = torch.tensor(embeddings, dtype=torch.float32, device=device).detach()
    labels = labels - torch.ones(labels.shape, dtype=torch.int32, device=device)
    labels = labels.type(torch.float32)
    labels = labels.to(device)
    train_mask = torch.tensor(([True] * 10000 + [False] * 5000), device=device)
    valid_mask = torch.tensor(([False] * 10000 + [True] * 2000 + [False] * 3000), device=device)
    test_mask = torch.tensor(([False] * 12000 + [True] * 3000), device=device)

    
    print("TRAIN")
    best_val = 0
    best_test = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        predictions = model(graph, x_word)

        loss = criteria(predictions[train_mask], labels[train_mask].unsqueeze(1).detach())
        
        loss.backward()
        optimizer.step()
        pred_val = predictions[valid_mask] >= 0.5
        pred_test = predictions[test_mask] >= 0.5

        res_val = torch.sum(pred_val.squeeze(1) == labels[valid_mask]).item()/2000
        res_test = torch.sum(pred_test.squeeze(1) == labels[test_mask]).item()/3000

        if res_val > best_val:
            best_test = res_test
            print("EPOCH: ", epoch)
            print("VAL: ", res_val)
            print("TEST: ", res_test)
            print("BEST_TEST", best_test)

            best_val = res_val
        
def preprocess_sentence(sentence):
    words = word_tokenize(sentence)
    words = [word.lower() for word in words]
    words = [word for word in words if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return words

def baseline(path="./data/train1.csv"):
    df = pd.read_csv(path)
    documents = []
    preprocessed_sentences = []
    labels = []
    embedds = []

    for i in range(15000):
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
    x = np.zeros((15000, 256))
    print("START")
    for i in range(15000):
        doc = documents[i]
        
        for sent in doc:
            for word in sent:
                x[i] += model.wv[word]
    print("END")

    train_x = x[:10000]
    valid_x = x[10000:12000]
    test_x = x[12000:]
    train_y = labels[:10000]
    valid_y = labels[10000:12000]
    test_y = labels[12000:]
    from sklearn.metrics import accuracy_score
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42)
    model.fit(train_x, train_y)
    preds = model.predict(test_x)

    print("ACC: ", accuracy_score(preds, test_y))


if __name__ == "__main__":
    # labels, graph = preprocess("./data/train1.csv")
    # labels, graph = build_graph()
    
    baseline()
    #train(graph, labels)