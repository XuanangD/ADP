import torch
import torch.nn.functional as F
import copy
import numpy as np
import matplotlib.pyplot as plt
from evaluate import evaluate
from multiDAE import MultiDAE
from multiVAE import MultiVAE
from read_data import read_ml1m, read_lastfm, read_steam, read_amazon, get_matrix
from evaluate import Metrics

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_set_dict, test_set_dict, num_user, num_item = read_ml1m('dataset/ml-1m/ratings.dat')
# train_set_dict, test_set_dict, num_user, num_item = read_lastfm('dataset/hetrec2011-lastfm-2k/user_artists.dat')
# train_set_dict, test_set_dict, num_user, num_item = read_steam('dataset/steam/steam-200k.csv')
train_set, test_set = get_matrix(train_set_dict, test_set_dict, nb_user=num_user, nb_item=num_item)
train_set = torch.FloatTensor(train_set).to(device)
# test_set = torch.FloatTensor(test_set).to(device)
model = MultiDAE(dec_dims=[200, 600, num_item],enc_dims=None, dropout=0.5, regs=0.01).to(device)
# model = MultiVAE(dec_dims=[200, 600, num_item],enc_dims=None, dropout=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
res_dict = {}
train_batch = 250

for e in range(100):
    print('Epoch %d' % (e+1))
    model.train()
    for batch in range(int(num_user/train_batch)):

        optimizer.zero_grad()
        train_index = torch.randint(0, num_user, (train_batch,)).to(device)
        data = train_set.index_select(0, train_index)
        recon_x = model(data)
        loss = model.loss(recon_x, data)
        # recon_batch, mu, logvar = model(train_set)
        # loss = model.loss(recon_batch, train_set, mu, logvar, beta=0.2)
        loss.backward()
        optimizer.step()

    recon_batch = model(train_set)
    loss = model.loss(recon_batch, train_set)
    # recon_batch, mu, logvar = model(train_set)
    # loss = model.loss(recon_batch, train_set, mu, logvar, beta=0.2)
    print("Loss: ", loss.item())
    model.eval()
    with torch.no_grad():
        recon_x = model(train_set)
        # recon_x, mu, logvar = model(train_set)
        recon_x[tuple(train_set.nonzero().t())] = -np.inf

    res = Metrics.compute(recon_x.cpu().numpy(), test_set, ["ndcg@100", "recall@100", "ndcg@20", "recall@20"])
    for k, v in res.items():
        res[k] = np.mean(np.nan_to_num(v))
        res_dict.setdefault(k, []).append(res[k])
    print(res)

