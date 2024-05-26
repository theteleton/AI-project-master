import torch
import dgl

from parameters import args
class GAT(torch.nn.Module):
    def __init__(self, in_dim=256, hid_dim=[128], out_dim=128, heads=1, dropout=0.5, device=torch.device("cpu")):
        super(GAT, self).__init__()
        self.dims = [in_dim] + hid_dim + [out_dim]
        self.convs = []
        self.dropout = dropout
        for in_f, in_out in zip(self.dims[:-1], self.dims[1:]):
            self.convs.append(dgl.nn.HeteroGraphConv({
                "word2word" : dgl.nn.GATConv(in_f, in_out, heads),
                "word2document" : dgl.nn.GATConv(in_f, in_out, heads),
                "word2wordr" : dgl.nn.GATConv(in_f, in_out, heads),
                "word2documentr" : dgl.nn.GATConv(in_f, in_out, heads)
            }, aggregate="stack").to(device))
        
        self.lin1 = torch.nn.Linear(out_dim, 1, device=device)
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.lin1.weight)
        torch.nn.init.ones_(self.lin1.bias)
    
    def forward(self, graph, x):
        for i, conv in enumerate(self.convs):
            # print("BEFORE: ", i," ", torch.sum(x["word"]))
            x = conv(graph, x)
            # print("AFTER: ", i, " ", torch.sum(x["word"]))
            for type in ["word", "document"]:
                if type not in x.keys():
                    continue
                x[type] = torch.sum(x[type], dim=1)
                x[type] = torch.sum(x[type], dim=1)
                x[type] = x[type].reshape((-1, self.dims[i + 1]))
            for type in ["word", "document"]:
                x[type] = torch.nn.functional.relu(x[type])
                
            
        for type in ["word", "document"]:
            if type not in x.keys():
                continue

            x[type] = self.lin1(x[type])
            x[type] = torch.nn.functional.sigmoid(x[type])
        
        return x
    
class GCN(torch.nn.Module):
    def __init__(self, in_dim=256, hid_dim=[128], out_dim=128, dropout=0.5, device=torch.device("cpu")):
        super(GCN, self).__init__()
        self.dims = [in_dim] + hid_dim + [out_dim]
        self.convs = []
        self.dropout = dropout
        for in_f, in_out in zip(self.dims[:-1], self.dims[1:]):
            self.convs.append(dgl.nn.HeteroGraphConv({
                "word2word" : dgl.nn.GraphConv(in_f, in_out),
                "word2document" : dgl.nn.GraphConv(in_f, in_out),
                "word2wordr" : dgl.nn.GraphConv(in_f, in_out),
                "word2documentr" : dgl.nn.GraphConv(in_f, in_out)
            }, aggregate="stack").to(device))
        
        self.lin1 = torch.nn.Linear(out_dim, 1, device=device)
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.lin1.weight)
        torch.nn.init.ones_(self.lin1.bias)
    
    def forward(self, graph, x):
        for i, conv in enumerate(self.convs):
            x = conv(graph, x)
            for type in ["word", "document"]:
                if type not in x.keys():
                    continue
                x[type] = torch.sum(x[type], dim=1)
                x[type] = x[type].reshape((-1, self.dims[i + 1]))
            for type in ["word", "document"]:
                x[type] = torch.nn.functional.relu(x[type])
                # x[type] = torch.nn.functional.dropout(x[type], p=self.dropout)
            
        for type in ["word", "document"]:
            if type not in x.keys():
                continue

            x[type] = self.lin1(x[type])
            x[type] = torch.nn.functional.sigmoid(x[type])
        return x

class SAGE(torch.nn.Module):
    def __init__(self, in_dim=256, hid_dim=[128], out_dim=128, dropout=0.5, device=torch.device("cpu")):
        super(SAGE, self).__init__()
        self.dims = [in_dim] + hid_dim + [out_dim]
        self.convs = []
        self.dropout = dropout
        for in_f, in_out in zip(self.dims[:-1], self.dims[1:]):
            self.convs.append(dgl.nn.HeteroGraphConv({
                "word2word" : dgl.nn.SAGEConv(in_f, in_out, "mean"),
                "word2document" : dgl.nn.SAGEConv(in_f, in_out, "mean"),
                "word2wordr" : dgl.nn.SAGEConv(in_f, in_out, "mean"),
                "word2documentr" : dgl.nn.SAGEConv(in_f, in_out, "mean")
            }, aggregate="stack").to(device))
        
        self.lin1 = torch.nn.Linear(out_dim, 1, device=device)
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.lin1.weight)
        torch.nn.init.ones_(self.lin1.bias)
    
    def forward(self, graph, x):
        for i, conv in enumerate(self.convs):
            x = conv(graph, x)
            for type in ["word", "document"]:
                if type not in x.keys():
                    continue
                x[type] = torch.sum(x[type], dim=1)

                x[type] = x[type].reshape((-1, self.dims[i + 1]))
            for type in ["word", "document"]:
                x[type] = torch.nn.functional.relu(x[type])
                
            
        for type in ["word", "document"]:
            if type not in x.keys():
                continue

            x[type] = self.lin1(x[type])
            x[type] = torch.nn.functional.sigmoid(x[type])
        return x

class Model(torch.nn.Module):
    def __init__(self, in_dim=256, h_dim=[128], out_dim=128, model="GAT", heads=4, dropout=0.5, device=torch.device("cpu")):
        super(Model, self).__init__()

        self.document_embedding = torch.nn.Embedding(15000, in_dim, device=device)
        
        if model == "GAT":
            self.model = GAT(in_dim, h_dim, out_dim, heads=heads, dropout=dropout, device=device)
        elif model == "GCN":
            self.model = GCN(in_dim, h_dim, out_dim, dropout, device)
        else:
            self.model = SAGE(in_dim, h_dim, out_dim, dropout, device)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.document_embedding.weight)

    def forward(self, graph, x):
        x_dict = {
            "word" : x,
            "document" : self.document_embedding.weight
        }
        x = self.model(graph, x_dict)
        return x["document"]
    