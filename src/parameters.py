import argparse

def parseArgs():
    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--gnn", default="SAGE", type=str)
    parser.add_argument("--embeddings", default="word2vec", type=str)
    parser.add_argument("--heads", default=2, type=int)
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--in_dim", default=256, type=int)
    parser.add_argument("--h_dim", default=[128], type=list_of_ints)
    parser.add_argument("--o_dim", default=128, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--wd", default=0.0005, type=float)
    parser.add_argument("--window", default=5, type=int)

    return parser.parse_args()
args = parseArgs()


