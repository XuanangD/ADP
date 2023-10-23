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

def ldp(data, sensitivity, epsilon):
    tensor = torch.clamp(data, min=-sensitivity, max=sensitivity)
    loc = torch.zeros_like(tensor)
    scale = torch.ones_like(tensor) * (2 * sensitivity/epsilon)
    tensor = tensor + torch.distributions.laplace.Laplace(loc, scale).sample().to(device)
    return tensor

train_set_dict, test_set_dict, num_user, num_item = read_ml1m('dataset/ml-1m/ratings.dat')
# train_set_dict, test_set_dict, num_user, num_item = read_lastfm('dataset/hetrec2011-lastfm-2k/user_artists.dat')
# train_set_dict, test_set_dict, num_user, num_item = read_steam('dataset/steam/steam-200k.csv')
train_set, test_set = get_matrix(train_set_dict, test_set_dict, nb_user=num_user, nb_item=num_item)
train_set = torch.FloatTensor(train_set).to(device)
# test_set = torch.FloatTensor(test_set).to(device)
# model = MultiDAE(dec_dims=[200, 600, num_item],enc_dims=None, dropout=0.5, regs=0.01).to(device)
model = MultiVAE(dec_dims=[200, 600, num_item],enc_dims=None, dropout=0.5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
res_dict = {}
train_batch= 250
sensitivity = 0.5
epsilon = 20

for epoch in range(400):
    print('Epoch %d' % (epoch+1))

    for batch in range(int(num_user/train_batch)):
        optimizer.zero_grad()
        norm_sum = [torch.norm(par.data, p=2).cpu().numpy() for par in model.parameters()][::2]
        importance = norm_sum / np.sum(norm_sum)
        budget = []
        for im in importance:
            budget.extend([epsilon * im] * 2)
        for u in np.random.choice(range(num_user), train_batch):
            data = train_set[u:u+1, :]
            loc_model = copy.deepcopy(model)

            loc_model.train()
            # recon_x = loc_model(data)
            # loss = loc_model.loss(recon_x, data)
            recon_x, mu, logvar = loc_model(data)
            loss = loc_model.loss(recon_x, data, mu, logvar, beta=0.2)
            loss.backward()

            for pa, pb, e in zip(loc_model.parameters(), model.parameters(), budget):
                pb.grad = ldp(pa.grad, sensitivity, e) + (pb.grad if pb.grad is not None else 0)

        for p in model.parameters():
            p.grad /= train_batch

        optimizer.step()

    # recon_batch = model(train_set)
    # loss = model.loss(recon_batch, train_set)
    recon_batch, mu, logvar = model(train_set)
    loss = model.loss(recon_batch, train_set, mu, logvar, beta=0.2)
    print("Loss: ", loss.item())

    model.eval()
    with torch.no_grad():
        # recon_x = model(train_set)
        recon_x, mu, logvar = model(train_set)
        recon_x[tuple(train_set.nonzero().t())] = -np.inf

    res = Metrics.compute(recon_x.cpu().numpy(), test_set,["ndcg@100", "recall@100", "ndcg@20", "recall@20"])
    for k, v in res.items():
        res[k] = np.mean(np.nan_to_num(v))
        res_dict.setdefault(k, []).append(res[k])
    print(res)

for k,v in res_dict.items():
    print(k, ','.join(map(str, v)))

# fig, ax = plt.subplots(1,4)
for i, k in zip(range(4), res_dict.keys()):
    plt.subplot(2, 2, i+1)
    plt.plot(range(len(res_dict[k])), res_dict[k])
    plt.title(k)

plt.tight_layout()
plt.show()
