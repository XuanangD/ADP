import torch
import torch.nn as nn
import torch.nn.functional as F
from multiDAE import MultiDAE


class MultiVAE(MultiDAE):
    def __init__(self, dec_dims, enc_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__(dec_dims, enc_dims)
        # Last dimension of enc- network is for mean and variance
        temp_dims = self.enc_dims[:-1] + [self.enc_dims[-1] * 2]
        self.encoder = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(temp_dims[:-1], temp_dims[1:])])

        self.decoder = nn.ModuleList(
            [nn.Linear(d_in, d_out) for d_in, d_out in zip(self.dec_dims[:-1], self.dec_dims[1:])])

        self.dropout = nn.Dropout(dropout)
        super().init_weight()

    def encode(self, x):
        """Apply the encoder network of the Variational Autoencoder.
        Parameters
        ----------
        x : :py:class:`torch.Tensor`
            The input tensor
        Returns
        -------
        mu, logvar : :obj:`tuple` of :py:class:`torch.Tensor`
            The tensors in the latent space representing the mean and standard deviation (actually
            the logarithm of the variance) of the probability distributions over the
            latent variables.
        """
        h = F.normalize(x)
        if self.training:
            h = self.dropout(h)
        for i, layer in enumerate(self.encoder):
            h = layer(h)
            if i != len(self.encoder) - 1:
                h = torch.tanh(h)
            else:
                mu = h[:, :self.enc_dims[-1]]
                logvar = h[:, self.enc_dims[-1]:]
        return mu, logvar

    def decode(self, z):
        """Apply the decoder network to the sampled latent representation.
        Parameters
        ----------
        z : :py:class:`torch.Tensor`
            The sampled (trhough the reparameterization trick) latent tensor.
        Returns
        -------
        :py:class:`torch.Tensor`
            The output tensor of the decoder network.
        """
        h = z
        for _, layer in enumerate(self.decoder[:-1]):
            h = torch.tanh(layer(h))
        return self.decoder[-1](h)

    def _reparameterize(self, mu, var):
        if self.training:
            std = torch.exp(0.5 * var)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        """Apply the full Variational Autoencoder network to the input.
        Parameters
        ----------
        x : :class:`torch.Tensor`
            The input tensor
        Returns
        -------
        x', mu, logvar : :obj:`tuple` of :py:class:`torch.Tensor`
            The reconstructed input (x') along with the intermediate tensors in the latent space
            representing the mean and standard deviation (actually the logarithm of the variance)
            of the probability distributions over the latent variables.
        """

        mu, logvar = self.encode(x)
        z = self._reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, recon_x, x, mu, logvar, beta=1.0):
        r"""VAE for collaborative filtering loss function.
               MultiVAE assumes a multinomial distribution over the input and this is reflected in the loss
               function. The loss is a :math:`\beta` ELBO (Evidence Lower BOund) in which the
               regularization part is weighted by a hyper-parameter :math:`\beta`. Moreover, as in
               MultiDAE, the reconstruction loss is based on the multinomial likelihood.
               Specifically, the loss function of MultiVAE is defined as:
               :math:`\mathcal{L}_{\beta}(\mathbf{x}_{u} ; \theta, \phi)=\
               \mathbb{E}_{q_{\phi}(\mathbf{z}_{u} | \mathbf{x}_{u})}[\log p_{\theta}\
               (\mathbf{x}_{u} | \mathbf{z}_{u})]-\beta \cdot \operatorname{KL}(q_{\phi}\
               (\mathbf{z}_{u} | \mathbf{x}_{u}) \| p(\mathbf{z}_{u}))`
               Parameters
               ----------
               recon_x : :class:`torch.Tensor`
                   The reconstructed input, i.e., the output of the variational autoencoder. It is meant
                   to be the reconstruction over a batch.
               x : :class:`torch.Tensor`
                   The input, and hence the target tensor. It is meant to be a batch size input.
               mu : :class:`torch.Tensor`
                   The mean part of latent space for the given ``x``. Together with ``logvar`` represents
                   the representation of the input ``x`` before the reparameteriation trick.
               logvar : :class:`torch.Tensor`
                   The (logarithm of the) variance part of latent space for the given ``x``. Together with
                   ``mu`` represents the representation of the input ``x`` before the reparameteriation
                   trick.
               beta : :obj:`float` [optional]
                   The current :math:`\beta` regularization hyper-parameter, by default 1.0.
               Returns
               -------
               :class:`torch.Tensor`
                   Tensor (:math:`1 \times 1`) representing the average loss incurred over the input
                   batch.
               """
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return BCE + beta * KLD