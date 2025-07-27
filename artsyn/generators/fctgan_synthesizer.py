import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, Conv2d, ConvTranspose2d, Sigmoid,
                      init, BCELoss, CrossEntropyLoss, SmoothL1Loss, GELU, LayerNorm, Identity, ModuleList, Parameter)
from artsyn.generators.fctgan_transformer import ImageTransformer, DataTransformer
from artsyn.generators.fno import FNO1d

from tqdm import tqdm
from timm.models.layers import DropPath, to_2tuple
import csv
import math


class Classifier(Module):
    def __init__(self, input_dim, dis_dims, st_ed):
        super(Classifier, self).__init__()
        dim = input_dim - (st_ed[1] - st_ed[0])
        seq = []
        self.str_end = st_ed
        for item in list(dis_dims):
            seq += [
                Linear(dim, item),
                LeakyReLU(0.2),
                Dropout(0.5)
            ]
            dim = item

        if (st_ed[1] - st_ed[0]) == 1:
            seq += [Linear(dim, 1)]
        elif (st_ed[1] - st_ed[0]) == 2:
            seq += [Linear(dim, 1), Sigmoid()]
        else:
            seq += [Linear(dim, (st_ed[1] - st_ed[0]))]

        self.seq = Sequential(*seq)

    def forward(self, inp):
        if (self.str_end[1] - self.str_end[0]) == 1:
            label = inp[:, self.str_end[0]:self.str_end[1]]
        else:
            label = torch.argmax(inp[:, self.str_end[0]:self.str_end[1]], axis=-1)

        new_imp = torch.cat((inp[:, :self.str_end[0]], inp[:, self.str_end[1]:]), 1)

        if ((self.str_end[1] - self.str_end[0]) == 2) | ((self.str_end[1] - self.str_end[0]) == 1):
            return self.seq(new_imp).view(-1), label
        else:
            return self.seq(new_imp), label


def apply_activate(data, output_info):
    data_t = []
    st = 0
    for item in output_info:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
    return torch.cat(data_t, dim=1)


def get_st_ed(target_col_index, output_info):
    st = 0
    c = 0
    tc = 0

    for item in output_info:
        if c == target_col_index:
            break
        if item[1] == 'tanh':
            st += item[0]
            if item[2] == 'yes_g':
                c += 1
        elif item[1] == 'softmax':
            st += item[0]
            c += 1
        tc += 1

    ed = st + output_info[tc][0]

    return st, ed


def random_choice_prob_index_sampling(probs, col_idx):
    option_list = []
    for i in col_idx:
        pp = probs[i]
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))

    return np.array(option_list).reshape(col_idx.shape)


def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)


def maximum_interval(output_info):
    max_interval = 0
    for item in output_info:
        max_interval = max(max_interval, item[0])
    return max_interval


class Cond(object):
    def __init__(self, data, output_info):

        self.model = []
        st = 0
        counter = 0
        for item in output_info:

            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                counter += 1
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                st = ed

        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        st = 0
        self.p = np.zeros((counter, maximum_interval(output_info)))
        self.p_sampling = []
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                tmp = np.sum(data[:, st:ed], axis=0)
                tmp_sampling = np.sum(data[:, st:ed], axis=0)
                tmp = np.log(tmp + 1)
                tmp = tmp / np.sum(tmp)
                tmp_sampling = tmp_sampling / np.sum(tmp_sampling)
                self.p_sampling.append(tmp_sampling)
                self.p[self.n_col, :item[0]] = tmp
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
                st = ed

        self.interval = np.asarray(self.interval)

    def sample_train(self, batch):
        if self.n_col == 0:
            return None
        batch = batch

        idx = np.random.choice(np.arange(self.n_col), batch)

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        mask = np.zeros((batch, self.n_col), dtype='float32')
        mask[np.arange(batch), idx] = 1
        opt1prime = random_choice_prob_index(self.p[idx])
        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1

        return vec, mask, idx, opt1prime

    def sample(self, batch):
        if self.n_col == 0:
            return None
        batch = batch

        idx = np.random.choice(np.arange(self.n_col), batch)

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        opt1prime = random_choice_prob_index_sampling(self.p_sampling, idx)

        for i in np.arange(batch):
            vec[i, self.interval[idx[i], 0] + opt1prime[i]] = 1

        return vec


def cond_loss(data, output_info, c, m):
    loss = []
    st = 0
    st_c = 0
    for item in output_info:
        if item[1] == 'tanh':
            st += item[0]
            continue

        elif item[1] == 'softmax':
            ed = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(
                data[:, st:ed],
                torch.argmax(c[:, st_c:ed_c], dim=1),
                reduction='none')
            loss.append(tmp)
            st = ed
            st_c = ed_c

    loss = torch.stack(loss, dim=1)
    return (loss * m).sum() / data.size()[0]


class Sampler(object):
    def __init__(self, data, output_info):
        super(Sampler, self).__init__()
        self.data = data
        self.model = []
        self.n = len(data)
        st = 0
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, st + j])[0])
                self.model.append(tmp)
                st = ed

    def sample(self, n, col, opt):
        if col is None:
            idx = np.random.choice(np.arange(self.n), n)
            return self.data[idx]
        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self.model[c][o]))
        return self.data[idx]


class Discriminator(Module):
    def __init__(self, side, layers, fno: int):
        super(Discriminator, self).__init__()
        self.side = side
        self.fno = fno
        info = len(layers) - (1 if self.fno == 1 else 2)
        self.seq = Sequential(*layers)
        self.seq_info = Sequential(*layers[:info])

    def forward(self, inp):
        return (self.seq(inp)), self.seq_info(inp)


class Mlp(Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = torch.nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(hidden_features, out_features)
        self.drop = torch.nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalFilter(Module):
    def __init__(self, dim, side):
        super().__init__()
        self.w = side
        self.h = int(side / 2) + 1
        self.complex_weight = Parameter(torch.randn(self.w, self.h, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x, spatial_size=None):
        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size

        x = x.view(B, a, b, C)

        x = x.to(torch.float32)
        # print("x shape before rfft2: ", x.shape)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        # print("x shape after rfft2: ", x.shape)
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x


class PatchEmbed(Module):
    """ 
    Image to Patch Embedding
    """

    def __init__(self, img_size, patch_size, in_chans=1, embed_dim=256):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, H, W, C = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x.permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)
        return x


class Block(Module):

    def __init__(self, side, dim, drop_path=0.2, dropout=0.2, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = LayerNorm(dim)
        self.filter = GlobalFilter(dim, side)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else Identity()
        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=dropout)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


class CFNO2d(Module):
    def __init__(self, side, dim, drop_path=0.2, dropout=0.2, mlp_ratio=4.0, depth=4, embed_dim=256):
        """
        dim here is actually the number of channels
        """
        super().__init__()
        print("side: ", side)
        self.img_size = side

        # Here the patch_size setting is important.
        # We know for exmaple Adult dataset is wrapped as 24*24, with patch_size = 8, there will have 9 (3*3) patches
        # after. The image size, patch size will influence the setting in GFGenerator() function. Unfortunately, user
        # may need to manually find the parameters for their own datasets.
        self.patch_size = 8
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = Dropout(p=dropout)
        self.head = Linear(dim, 1)
        if dropout > 0:
            self.final_dropout = Dropout(dropout)
        else:
            self.final_dropout = Identity()
        self.norm = LayerNorm(dim)

        self.blocks = ModuleList(
            [Block(self.img_size // self.patch_size, dim, drop_path, dropout, mlp_ratio) for _ in range(depth)])

    def forward(self, x):

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x).mean(1)
        x = self.final_dropout(x)
        x = self.head(x)

        return x


class matmul(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        x = x1 @ x2
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def bicubic_upsample(x, H, W):
    """
    Used to upscale image size
    """
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bicubic')
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


def pixel_upsample(x, H, W):
    """
    Used to upscale image size
    """
    B, N, C = x.size()
    assert N == H * W
    x = x.permute(0, 2, 1)
    x = x.view(-1, C, H, W)
    x = torch.nn.PixelShuffle(2)(x)
    B, C, H, W = x.size()
    x = x.view(-1, C, H * W)
    x = x.permute(0, 2, 1)
    return x, H, W


class BlockGenerator(Module):

    def __init__(self, side, dim=64, drop_path=0.2, dropout=0.2, mlp_ratio=4.0, output_features=-1):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.filter = GlobalFilter(dim, side)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else Identity()
        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       drop=dropout) if output_features == -1 else Mlp(in_features=dim,
                                                                       hidden_features=int(dim * mlp_ratio),
                                                                       out_features=output_features, drop=dropout)

    def forward(self, x):
        x = x + self.drop_path(self.mlp(self.norm2(self.filter(self.norm1(x)))))
        return x


class GFGenerator(Module):
    """
    Build whole generator with FNO blocks.
    """

    def __init__(self, gside, base_width=4, random_dim=100, embed_dim=256):
        super(GFGenerator, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.bottom_width = base_width
        self.latent_dim = random_dim
        self.embed_dim = embed_dim
        self.pos_embed_1 = torch.nn.Parameter(torch.zeros(1, self.bottom_width ** 2, embed_dim))
        self.pos_embed_2 = torch.nn.Parameter(torch.zeros(1, (self.bottom_width * 2) ** 2, embed_dim // 4))
        self.pos_embed_3 = torch.nn.Parameter(torch.zeros(1, (self.bottom_width * 4) ** 2, embed_dim // 16))
        self.pos_embed_4 = torch.nn.Parameter(torch.zeros(1, (self.bottom_width * 4) ** 2, embed_dim // 16))
        self.pos_embed = [
            self.pos_embed_1,
            self.pos_embed_2,
            self.pos_embed_3,
            self.pos_embed_4

        ]
        self.block_1 = BlockGenerator(side=base_width, dim=embed_dim)
        self.block_2 = BlockGenerator(side=base_width * 2, dim=embed_dim // 4)
        self.block_3 = BlockGenerator(side=base_width * 4, dim=embed_dim // 16)
        self.block_4 = BlockGenerator(side=base_width * 4, dim=embed_dim // 16)
        self.l1 = torch.nn.Linear(self.latent_dim, (self.bottom_width ** 2) * embed_dim)
        self.deconv = torch.nn.Sequential(
            torch.nn.Conv2d(self.embed_dim // 16, 1, (1, 1))
        )

    def forward(self, z):
        x = self.l1(torch.squeeze(z))
        x = x.view(-1, self.bottom_width ** 2, self.embed_dim) + self.pos_embed[0].to(self.device)

        H, W = self.bottom_width, self.bottom_width
        x = self.block_1(x)

        x, H, W = pixel_upsample(x, H, W)
        x = x + self.pos_embed[1].to(self.device)
        x = self.block_2(x)

        x, H, W = pixel_upsample(x, H, W)
        x = x + self.pos_embed[2].to(self.device)
        x = self.block_3(x)

        x = x + self.pos_embed[3].to(self.device)
        x = self.block_4(x)

        output = self.deconv(
            x.view(-1, self.bottom_width * 4, self.bottom_width * 4, self.embed_dim // 16).permute(0, 3, 1, 2))

        return output


class Generator(Module):
    def __init__(self, side, layers, fno: int):
        super(Generator, self).__init__()
        self.side = side
        self.fno = fno
        self.seq = Sequential(*layers)

    def forward(self, input_):
        return self.seq(input_)


def determine_layers_disc(side, num_channels, fno: int):
    layers_d = []
    if fno == 1:
        layers_d += [FNO1d(16, 64), Sigmoid()]
    elif fno == 2:
        print("use 2d FNO as discriminator!")
        layers_d += [CFNO2d(side, num_channels, drop_path=0.2, dropout=0.2, mlp_ratio=4,
                            depth=4)]  # no sigmoid here since we wgan+gp as loss for discriminator
    else:
        num_channels = 64
        print("use default CNN as discriminator!")
        assert side >= 4 and side <= 64

        layer_dims = [(1, side), (num_channels, side // 2)]

        while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
            layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

        layerNorms = []
        num_c = num_channels
        num_s = side / 2
        for l in range(len(layer_dims) - 1):
            layerNorms.append([int(num_c), int(num_s), int(num_s)])
            num_c = num_c * 2
            num_s = num_s / 2

        layers_d = []

        for prev, curr, ln in zip(layer_dims, layer_dims[1:], layerNorms):
            layers_d += [
                Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
                LayerNorm(ln),
                LeakyReLU(0.2, inplace=True),
            ]

        layers_d += [Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0), ReLU(True)]

    return layers_d


def determine_layers_gen(side, random_dim, num_channels, fno: int):
    layers_g = []
    if fno == 1:
        layers_g += [FNO1d(16, 64)]
    elif fno == 2:
        print("use 2d FNO as generator!")
        layers_g += [GFGenerator(side, random_dim=random_dim)]
        # layers_g += [GlobalFilter(num_channels, side, side)]
    else:
        num_channels = 64
        print("use default CNN as generator!")
        assert side >= 4 and side <= 64

        layer_dims = [(1, side), (num_channels, side // 2)]

        while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
            layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

        layerNorms = []

        num_c = num_channels * (2 ** (len(layer_dims) - 2))
        num_s = int(side / (2 ** (len(layer_dims) - 1)))
        for l in range(len(layer_dims) - 1):
            layerNorms.append([int(num_c), int(num_s), int(num_s)])
            num_c = num_c / 2
            num_s = num_s * 2

        layers_g = [
            ConvTranspose2d(random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)]

        for prev, curr, ln in zip(reversed(layer_dims), reversed(layer_dims[:-1]), layerNorms):
            layers_g += [LayerNorm(ln), ReLU(True),
                         ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True)]

    return layers_g


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1)).view(val.size(0), 1)
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high

    return res


def calc_gradient_penalty_slerp(netD, real_data, fake_data, transformer, device='cpu', lambda_=10):
    batchsize = real_data.shape[0]
    alpha = torch.rand(batchsize, 1, device=device)
    interpolates = slerp(alpha, real_data, fake_data)
    interpolates = interpolates.to(device)
    interpolates = transformer.transform(interpolates)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    disc_interpolates, _ = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean() * lambda_

    return gradient_penalty


def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)


class FCTGANSynthesizer:
    # Here we set gen_fno and disc_fno to 2 to use 2D FNO. If they are set to 0, they use the same generator and
    # discriminator structures as CTAB-GAN+
    def __init__(self, class_dim=(256, 256, 256, 256), random_dim=100, num_channels=256, l2scale=1e-5, batch_size=500,
                 epochs=150, gen_fno=2, disc_fno=2, random_state=None):

        self.gen_fno = gen_fno
        self.disc_fno = disc_fno
        self.random_dim = random_dim
        self.class_dim = class_dim
        self.num_channels = num_channels
        self.dside = None
        self.gside = None
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.transformer = None
        self.generator = None
        self.cond_generator = None
        self.g_transformer = None
        self.d_transformer = None

        self._random_state = random_state

    def fit(self, train_data=pd.DataFrame, categorical=[], mixed={}, general=[], non_categorical=[], p_type={},
            data_prep=None):

        loss_d_list = []
        loss_g_info = []
        loss_g_list = []
        problem_type = None
        target_index = None
        if p_type:
            problem_type = list(p_type.keys())[0]
            if problem_type:
                target_index = train_data.columns.get_loc(p_type[problem_type])

        self.transformer = DataTransformer(train_data=train_data, categorical_list=categorical, mixed_dict=mixed,
                                           general_list=general, non_categorical_list=non_categorical)
        self.transformer.fit()
        train_data = self.transformer.transform(train_data.values)

        print("training data shape: ", train_data.shape)

        data_sampler = Sampler(train_data, self.transformer.output_info)
        data_dim = self.transformer.output_dim
        self.cond_generator = Cond(train_data, self.transformer.output_info)

        sides = [4, 8, 16, 24, 32, 64]
        col_size_d = data_dim + self.cond_generator.n_opt
        for i in sides:
            if i * i >= col_size_d:
                self.dside = i
                break

        sides = [4, 8, 16, 24, 32, 64]
        col_size_g = data_dim
        for i in sides:
            if i * i >= col_size_g:
                self.gside = i
                break

        print("gside dside: ", self.gside, self.dside)

        layers_g = determine_layers_gen(self.gside, self.random_dim + self.cond_generator.n_opt, self.num_channels,
                                        fno=self.gen_fno)
        layers_d = determine_layers_disc(self.dside, self.num_channels, fno=self.disc_fno)
        self.generator = Generator(self.gside, layers_g, self.gen_fno).to(self.device)
        discriminator = Discriminator(self.dside, layers_d, self.disc_fno).to(self.device)

        fno_optimizer_params = dict(lr=1.5e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizer_params = dict(lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)

        params_g = fno_optimizer_params if self.gen_fno != 0 else optimizer_params
        params_d = fno_optimizer_params if self.disc_fno != 0 else optimizer_params

        optimizer_g = Adam(self.generator.parameters(), **params_g)
        optimizer_d = Adam(discriminator.parameters(), **params_d)
        scheduler_d = None
        if self.disc_fno != 0:
            scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=50, gamma=0.5)

        st_ed = None
        classifier = None
        optimizer_c = None
        if target_index is not None:
            st_ed = get_st_ed(target_index, self.transformer.output_info)
            classifier = Classifier(data_dim, self.class_dim, st_ed).to(self.device)
            optimizer_c = optim.Adam(classifier.parameters(), **optimizer_params)

        self.generator.apply(weights_init)
        discriminator.apply(weights_init)

        self.g_transformer = ImageTransformer(self.gside, self.gen_fno)
        self.d_transformer = ImageTransformer(self.dside, self.disc_fno)

        epsilon = 0
        epoch = 0
        steps = 0
        ci = 5

        steps_per_epoch = max(1, len(train_data) // self.batch_size)
        for i in tqdm(range(self.epochs)):

            for id_ in range(steps_per_epoch):
                for _ in range(ci):
                    noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                    condvec = self.cond_generator.sample_train(self.batch_size)

                    c, m, col, opt = condvec
                    c = torch.from_numpy(c).to(self.device)
                    m = torch.from_numpy(m).to(self.device)
                    noisez = torch.cat([noisez, c], dim=1)
                    noisez = noisez.view(self.batch_size, self.random_dim + self.cond_generator.n_opt, 1, 1)

                    perm = np.arange(self.batch_size)
                    np.random.shuffle(perm)
                    real = data_sampler.sample(self.batch_size, col[perm], opt[perm])
                    c_perm = c[perm]

                    real = torch.from_numpy(real.astype('float32')).to(self.device)

                    fake = self.generator(noisez)

                    faket = self.g_transformer.inverse_transform(fake)

                    fakeact = apply_activate(faket, self.transformer.output_info)

                    fake_cat = torch.cat([fakeact, c], dim=1)
                    real_cat = torch.cat([real, c_perm], dim=1)

                    real_cat_d = self.d_transformer.transform(real_cat)
                    fake_cat_d = self.d_transformer.transform(fake_cat)
                    optimizer_d.zero_grad()

                    d_real, _ = discriminator(real_cat_d)

                    d_real = -torch.mean(d_real)
                    d_real.backward()

                    d_fake, _ = discriminator(fake_cat_d)

                    d_fake = torch.mean(d_fake)

                    loss_d_list.append(d_fake.item())

                    d_fake.backward()

                    pen = calc_gradient_penalty_slerp(discriminator, real_cat, fake_cat, self.d_transformer, self.device)

                    pen.backward()

                    optimizer_d.step()

                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)

                condvec = self.cond_generator.sample_train(self.batch_size)

                c, m, col, opt = condvec
                c = torch.from_numpy(c).to(self.device)
                m = torch.from_numpy(m).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez = noisez.view(self.batch_size, self.random_dim + self.cond_generator.n_opt, 1, 1)

                optimizer_g.zero_grad()

                fake = self.generator(noisez)
                faket = self.g_transformer.inverse_transform(fake)
                fakeact = apply_activate(faket, self.transformer.output_info)

                fake_cat = torch.cat([fakeact, c], dim=1)
                fake_cat = self.d_transformer.transform(fake_cat)

                y_fake, info_fake = discriminator(fake_cat)

                cross_entropy = cond_loss(faket, self.transformer.output_info, c, m)

                _, info_real = discriminator(real_cat_d)

                g = -torch.mean(y_fake) + cross_entropy

                loss_g_list.append(g.item())

                g.backward(retain_graph=True)
                loss_mean = torch.norm(torch.mean(info_fake.view(self.batch_size, -1), dim=0) - torch.mean(
                    info_real.view(self.batch_size, -1), dim=0), 1)
                loss_std = torch.norm(torch.std(info_fake.view(self.batch_size, -1), dim=0) - torch.std(
                    info_real.view(self.batch_size, -1), dim=0), 1)
                loss_info = loss_mean + loss_std

                loss_g_info.append(loss_info.item())

                loss_info.backward()
                optimizer_g.step()

                if problem_type:

                    fake = self.generator(noisez)

                    faket = self.g_transformer.inverse_transform(fake)

                    fakeact = apply_activate(faket, self.transformer.output_info)

                    real_pre, real_label = classifier(real)
                    fake_pre, fake_label = classifier(fakeact)

                    c_loss = CrossEntropyLoss()

                    if (st_ed[1] - st_ed[0]) == 1:
                        c_loss = SmoothL1Loss()
                        real_label = real_label.type_as(real_pre)
                        fake_label = fake_label.type_as(fake_pre)
                        real_label = torch.reshape(real_label, real_pre.size())
                        fake_label = torch.reshape(fake_label, fake_pre.size())


                    elif (st_ed[1] - st_ed[0]) == 2:
                        c_loss = BCELoss()
                        real_label = real_label.type_as(real_pre)
                        fake_label = fake_label.type_as(fake_pre)

                    loss_cc = c_loss(real_pre, real_label)
                    loss_cg = c_loss(fake_pre, fake_label)

                    optimizer_g.zero_grad()
                    loss_cg.backward()
                    optimizer_g.step()

                    optimizer_c.zero_grad()
                    loss_cc.backward()
                    optimizer_c.step()

            if scheduler_d:
                scheduler_d.step()

            if (i + 1) % 50 == 0:
                sample = self.sample_fast(len(train_data))
                sample_df = data_prep.inverse_prep(sample)
                sample_df.to_csv("fake_{exp}.csv".format(exp=i), index=False)

        print(len(loss_d_list), len(loss_g_info), len(loss_g_list))
        rows = zip(loss_g_info, loss_g_list)
        with open("g_loss_record.csv", "w") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

        rows = zip(loss_d_list)
        with open("d_loss_record.csv", "w") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

    def sample(self, n):

        self.generator.eval()

        output_info = self.transformer.output_info
        steps = n // self.batch_size + 1

        data = []
        for i in range(steps):
            noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
            condvec = self.cond_generator.sample(self.batch_size)
            c = condvec
            c = torch.from_numpy(c).to(self.device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez = noisez.view(self.batch_size, self.random_dim + self.cond_generator.n_opt, 1, 1)

            fake = self.generator(noisez)
            faket = self.g_transformer.inverse_transform(fake)
            fakeact = apply_activate(faket, output_info)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        result, resample = self.transformer.inverse_transform(data)

        while len(result) < n:
            data_resample = []
            steps_left = resample // self.batch_size + 1

            for i in range(steps_left):
                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                condvec = self.cond_generator.sample(self.batch_size)
                c = condvec
                c = torch.from_numpy(c).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez = noisez.view(self.batch_size, self.random_dim + self.cond_generator.n_opt, 1, 1)

                fake = self.generator(noisez)
                faket = self.g_transformer.inverse_transform(fake)
                fakeact = apply_activate(faket, output_info)
                data_resample.append(fakeact.detach().cpu().numpy())

            data_resample = np.concatenate(data_resample, axis=0)

            res, resample = self.transformer.inverse_transform(data_resample)
            result = np.concatenate([result, res], axis=0)
            print("generated result length: ", result.shape)

        return result[0:n]

    def sample_fast(self, n):

        self.generator.eval()

        output_info = self.transformer.output_info
        steps = n // self.batch_size + 1

        data = []
        for i in range(steps):
            noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
            condvec = self.cond_generator.sample(self.batch_size)
            c = condvec
            c = torch.from_numpy(c).to(self.device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez = noisez.view(self.batch_size, self.random_dim + self.cond_generator.n_opt, 1, 1)

            fake = self.generator(noisez)
            faket = self.g_transformer.inverse_transform(fake)
            fakeact = apply_activate(faket, output_info)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        result, resample = self.transformer.inverse_transform_fast(data)

        while len(result) < n:
            data_resample = []
            steps_left = resample // self.batch_size + 1

            for i in range(steps_left):
                noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
                condvec = self.cond_generator.sample(self.batch_size)
                c = condvec
                c = torch.from_numpy(c).to(self.device)
                noisez = torch.cat([noisez, c], dim=1)
                noisez = noisez.view(self.batch_size, self.random_dim + self.cond_generator.n_opt, 1, 1)

                fake = self.generator(noisez)
                faket = self.g_transformer.inverse_transform(fake)
                fakeact = apply_activate(faket, output_info)
                data_resample.append(fakeact.detach().cpu().numpy())

            data_resample = np.concatenate(data_resample, axis=0)

            res, resample = self.transformer.inverse_transform_fast(data_resample)
            result = np.concatenate([result, res], axis=0)
            print("generated result length: ", result.shape)

        return result[0:n]
