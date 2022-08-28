import torch
import torch.nn.functional as F
from torch import nn
from ..utils import make_mlp
from ..VAEbase import VAEBase
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

class DownSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.Conv2d(dim, dim, 4, stride = 2, padding = 1)
        self.res = nn.Conv2d(dim, dim, 2, stride = 2)
        self.act = nn.GELU()

    def forward(self, x):
        
        h = self.pool(x)
        h = self.act(h) + self.res(x)
        
        return h

class UpSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool = nn.ConvTranspose2d(dim, dim, 4, stride = 2, padding = 1)
        self.res = nn.ConvTranspose2d(dim, dim, 2, stride = 2)
        self.act = nn.GELU()

    def forward(self, x):
        
        h = self.pool(x)
        h = self.act(h) + self.res(x)
        
        return h

class ConvNextBlock(nn.Module):

    def __init__(self, dim, dim_out, mult=2, norm=True):
        super().__init__()
        
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):

        h = self.ds_conv(x)
        h = self.net(h)
        h = h + self.res_conv(x)

        return h

class Encoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        in_dim = 1
        encoder_layers = []

        assert hparams["image_size"] % (2**len(hparams["blocks"])) == 0

        for num_blocks, num_channels in zip(hparams["blocks"], hparams["channels"]):

            encoder_layers.append(ConvNextBlock(in_dim, num_channels))
            in_dim = num_channels
            encoder_layers.extend([ConvNextBlock(in_dim, num_channels) for _ in range(num_blocks-1)])
            encoder_layers.append(DownSample(in_dim))
        
        encoder_layers.append(nn.Flatten())

        self.encoder = nn.Sequential(
           *encoder_layers 
        )

        style_emb_dim = hparams["channels"][-1] * (hparams["image_size"] // (2**len(hparams["blocks"])))**2

        self.encoder_mu_layers = make_mlp(
            style_emb_dim + hparams["content_emb_dim"],
            hparams["mlp_hidden"],
            style_emb_dim,
            hparams["mlp_layers"],
            hidden_activation="GELU",
            output_activation=None,
            layer_norm=True,
        )
        
        self.encoder_logvar_layers = make_mlp(
            style_emb_dim + hparams["content_emb_dim"],
            hparams["mlp_hidden"],
            style_emb_dim,
            hparams["mlp_layers"],
            hidden_activation="GELU",
            output_activation=None,
            layer_norm=True,
        )
        self.hparams = hparams

    def forward(self, imgs, contents):
        
        imgs = self.encoder(imgs).squeeze()
        z = torch.cat([imgs, contents], dim = 1)
        mu = self.encoder_mu_layers(z)
        logvar = self.encoder_logvar_layers(z)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        style_emb_dim = hparams["channels"][-1] * (hparams["image_size"] // (2**len(hparams["blocks"])))**2

        self.decoder_mlp_layers = make_mlp(
            style_emb_dim + hparams["content_emb_dim"],
            hparams["mlp_hidden"],
            style_emb_dim,
            hparams["mlp_layers"],
            hidden_activation="GELU",
            output_activation=None,
            layer_norm=True,
        )

        decoder_layers = [nn.PixelShuffle(hparams["image_size"] // (2**len(hparams["blocks"])))]
        in_dim = hparams["channels"][-1]

        for num_blocks, num_channels in zip(hparams["blocks"][::-1], hparams["channels"][::-1]):
            
            decoder_layers.append(UpSample(in_dim))
            decoder_layers.append(ConvNextBlock(in_dim, num_channels))
            in_dim = num_channels
            decoder_layers.extend([ConvNextBlock(in_dim, num_channels) for _ in range(num_blocks-1)])

        self.decoder = nn.Sequential(
           *decoder_layers 
        )

        self.decoder_1x1_layers = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1),
            nn.GELU(),
            nn.Conv2d(in_dim, in_dim, 1),
            nn.GELU(),
            nn.Conv2d(in_dim, 1, 1),
            nn.Tanh()
        )

        self.hparams = hparams

    def forward(self, z, contents):
    
        z = torch.cat([z, contents], dim = 1)
        z = self.decoder_mlp_layers(z).unsqueeze(2).unsqueeze(3)
        imgs = self.decoder(z)
        imgs = self.decoder_1x1_layers(imgs)

        return imgs

class ZiCVAE(VAEBase):

    """
    An interaction network class
    """

    def __init__(self, hparams):
        super().__init__(hparams) 
    
        self.content_embeddings = nn.Embedding(
            self.hparams["supports"],
            self.hparams["content_emb_dim"]
        )

        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)

    def encode(self, imgs, in_idxs):
        return self.encoder(imgs, self.content_embeddings(in_idxs))        

    def decode(self, z, out_idxs):    
        return self.decoder(z, self.content_embeddings(out_idxs))

    def forward(self, imgs, in_idxs, out_idxs):

        in_idxs, out_idxs = in_idxs.squeeze(), out_idxs.squeeze()

        mu, logvar = self.encode(imgs, in_idxs)
        z = self.reparameterize(mu, logvar) if self.training else mu
        imgs = self.decode(z, out_idxs)
        imgs = torch.tanh(imgs)

        return imgs, mu, logvar