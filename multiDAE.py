import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiDAE(nn.Module):
    def __init__(self, dec_dims, enc_dims=None, dropout=0.5, regs=0.01):
        super(MultiDAE, self).__init__()
        self.dec_dims = dec_dims
        if enc_dims is None:
            self.enc_dims = dec_dims[::-1]
        else:
            self.enc_dims = enc_dims

        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(self.enc_dims[:-1], self.enc_dims[1:])])
        self.decoder = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(self.dec_dims[:-1], self.dec_dims[1:])])
        self.regs = regs

        self.init_weight()

    def init_weight(self):
        """Initialize the weights of the network.
           Weights are initialized with the :py:func:`torch.nn.init.xavier_uniform_` initializer,
           while biases are initalized with the :py:func:`torch.nn.init.normal_` initializer.
        """
        for layer in self.encoder:
            nn.init.xavier_normal_(layer.weight)
            nn.init.normal_(layer.bias)
        for layer in self.decoder:
            nn.init.xavier_normal_(layer.weight)
            nn.init.normal_(layer.bias)

    def encode(self, x):
        """Forward propagate the input in the encoder network.
           Parameters
           ----------
           x : :py:class:`torch.Tensor`
               The input tensor
        """
        h = F.normalize(x)
        if self.training:
            h = self.dropout(h)
        for _, layer in enumerate(self.encoder):
            h = torch.tanh(layer(h))
        return h

    def decode(self, z):
        """Forward propagate the latent represenation in the decoder network.
           Parameters
           ----------
           z : :py:class:`torch.Tensor`
               The latent tensor
        """
        h = z
        for i, layer in enumerate(self.decoder):
            h = layer(h)
            if i != len(self.decoder) - 1:
                h = torch.tanh(h)
        return h

    def forward(self, x):
        """Forward propagate the input in the network.
           Parameters
           ----------
           x : :py:class:`torch.Tensor`
               The input tensor to feed to the network.
        """
        z = self.encode(x)
        return self.decode(z)

    def loss(self, recon_x, x):
        """Multinomial likelihood denoising autoencoder loss.
           Since the model assume a multinomial distribution over the input, then the reconstruction
           loss must be modified with respect to a vanilla VAE. In particular,
           the MultiDAE loss function is a combination of a reconstruction loss and a regularization
           loss, i.e.,
           :math:`\mathcal{L}(\mathbf{x}_{u} ; \\theta, \phi) =\
           \mathcal{L}_{rec}(\mathbf{x}_{u} ; \\theta, \phi) + \lambda\
           \mathcal{L}_{reg}(\mathbf{x}_{u} ; \\theta, \phi)`
           where
           :math:`\mathcal{L}_{rec}(\mathbf{x}_{u} ; \\theta, \phi) =\
           \mathbb{E}_{q_{\phi}(\mathbf{z}_{u} | \mathbf{x}_{u})}[\log p_{\\theta}\
           (\mathbf{x}_{u} | \mathbf{z}_{u})]`
           and
           :math:`\mathcal{L}_{reg}(\mathbf{x}_{u} ; \\theta, \phi) = \| \\theta \|_2 + \| \phi \|_2`,
           with :math:`\mathbf{x}_u` the input vector and :math:`\mathbf{z}_u` the latent vector
           representing the user *u*.
           Parameters
           ----------
           recon_x : :class:`torch.Tensor`
               The reconstructed input, i.e., the output of the variational autoencoder. It is meant
               to be the reconstruction over a batch.
           x : :class:`torch.Tensor`
               The input, and hence the target tensor. It is meant to be a batch size input.
           Returns
           -------
           :class:`torch.Tensor`
               Tensor (:math:`1 \times 1`) representing the average loss incurred over the input
               batch.
        """
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        l2_reg = 0
        for W in self.parameters():
            l2_reg += W.norm(2)

        return BCE + self.regs * l2_reg




