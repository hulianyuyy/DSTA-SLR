from functools import partial
from unittest.mock import patch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from model.attention import MHSA, DepthWiseConv2d, Mlp, bn_init, conv_init, import_class
from model.dropSke import DropBlock_Ske
from model.dropT import DropBlockT_1d
from timm.models.layers import DropPath, trunc_normal_
from einops import rearrange


class unit_san(nn.Module):
    def __init__(
        self, in_channels, out_channels, A, groups, num_point, num_subset=3, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_point = num_point
        self.groups = groups
        self.num_subset = num_subset
        num_subset = 8
        self.num_subset = num_subset
        self.inter_channels = out_channels // num_subset
        self.attention0s = nn.Parameter(
            torch.ones(1, num_subset, self.inter_channels, num_point, num_point)
            / num_point,
            requires_grad=True,
        )

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        # self.spatial_pos_embed_layer = nn.Parameter(torch.zeros(1, out_channels, 1, num_point))
        # trunc_normal_(self.spatial_pos_embed_layer, std=.02)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        """self.attention_block = attn_block(
                in_channels, 
                out_channels, 
                num_heads=s_num_heads, 
                mlp_ratio=4., 
                qkv_bias=False, 
                qk_scale=None, 
                attn_drop=0.0,
                drop=0., 
                drop_path=0., 
                act_layer=nn.GELU, 
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                use_grpe=use_grpe, 
                s_pos_emb=s_pos_emb,
                num_point= num_point,
            )"""

        self.inter_channels = out_channels // num_subset
        self.in_nets = nn.Conv2d(
            out_channels, 2 * num_subset * self.inter_channels, 1, bias=True
        )
        self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
        self.bn0 = nn.BatchNorm2d(self.inter_channels * num_subset)
        self.tan = nn.Tanh()
        self.norm1 = nn.LayerNorm(out_channels)
        # self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        mlp_ratio = 4
        mlp_hidden_dim = int(out_channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=out_channels,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.Linear_weight = nn.Parameter(
            torch.zeros(
                in_channels,
                self.inter_channels * num_subset,
                requires_grad=True,
                device="cuda",
            ),
            requires_grad=True,
        )
        nn.init.normal_(
            self.Linear_weight, 0, math.sqrt(0.5 / (self.inter_channels * num_subset))
        )

        self.Linear_bias = nn.Parameter(
            torch.zeros(
                1,
                self.inter_channels * num_subset,
                1,
                1,
                requires_grad=True,
                device="cuda",
            ),
            requires_grad=True,
        )
        nn.init.constant(self.Linear_bias, 1e-6)

        self.Edge_conv = Edge_feature_conv(out_channels)
        self.edge_weight = nn.Parameter(torch.zeros(1))

    def norm(self, A):
        b, c, h, w = A.size()
        A = A.view(c, self.num_point, self.num_point)
        D_list = torch.sum(A, 1).view(c, 1, self.num_point)
        D_list_12 = (D_list + 0.001) ** (-1)
        D_12 = self.eyes * D_list_12
        A = torch.bmm(A, D_12).view(b, c, h, w)
        return A

    def forward(self, x0):
        # x = self.attention_block(x0)

        x = torch.einsum("nctw,cd->ndtw", (x0, self.Linear_weight)).contiguous()
        x = x + self.Linear_bias
        x = self.bn0(x)

        n, c, t, v = x.size()
        x_resi = x
        edge_features = self.Edge_conv(x) * self.edge_weight
        # x = x + self.spatial_pos_embed_layer
        x = rearrange(x, "b c t v -> (b t) v c")
        x = self.norm1(x)
        x = rearrange(x, "(b t) v c -> b c t v", b=n)
        # x = self.norm1(x)
        q, k = torch.chunk(
            self.in_nets(x).view(n, 2 * self.num_subset, self.inter_channels, t, v),
            2,
            dim=1,
        )  # nctv -> n num_subset c'tv
        attention = (
            self.tan(
                torch.einsum("nkctu,nkctv->nkuv", [q, k]) / (self.inter_channels * t)
            )
            * self.alphas
        )
        x = x.view(n, self.num_subset, -1, t, v)
        x = torch.einsum("nkctv,nkvw->nkctw", (x, attention)).view(n, -1, t, v) + x_resi

        x_resi = x
        x = rearrange(x, "b c t v -> (b t) v c")
        x = self.norm2(x)
        x = x_resi + rearrange(self.mlp(x), "(b t) v c -> b c t v", b=n) + edge_features

        n, kc, t, v = x.size()
        x = x.view(n, self.num_subset, kc // self.num_subset, t, v)
        x = torch.einsum(
            "nkctv,nkcvw->nkctw", (x, self.attention0s.repeat(n, 1, 1, 1, 1))
        ).view(n, -1, t, v)
        # x = torch.einsum('nkctv,nkcvw->nctw', (x, self.attention0s.repeat(n,1,1,1,1)))

        x = self.bn(x)
        x += self.down(x0)
        x = self.relu(x)
        return x


class unit_tcn_skip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(unit_tcn_skip, self).__init__()

        self.pool = DepthWiseConv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=0,
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)

        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.pool(x))
        return x


class unit_tcn(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        stride=1,
        num_point=25,
        block_size=41,
    ):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

        self.dropS = DropBlock_Ske(num_point=num_point)
        self.dropT = DropBlockT_1d(block_size=block_size)

    def forward(self, x, keep_prob, A):
        x = self.bn(self.conv(x))
        x = self.dropT(self.dropS(x, keep_prob, A), keep_prob)
        return x


class UnfoldTemporalWindows(nn.Module):
    def __init__(self, window_size, window_stride, window_dilation=1):
        super().__init__()
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_dilation = window_dilation

        self.padding = (
            window_size + (window_size - 1) * (window_dilation - 1) - 1
        ) // 2
        self.unfold = nn.Unfold(
            kernel_size=(self.window_size, 1),
            dilation=(self.window_dilation, 1),
            stride=(self.window_stride, 1),
            padding=(self.padding, 0),
        )

    def forward(self, x):
        # Input shape: (N,C,T,V), out: (N,C,T,V*window_size)
        N, C, T, V = x.shape
        x = self.unfold(x)  # (N, C*Window_Size, (T-Window_Size+1)*(V-1+1))
        # Permute extra channels from window size to the graph dimension; -1 for number of windows
        x = x.view(N, C, self.window_size, -1, V).permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(N, C, -1, self.window_size, V)
        return x


class unit_tan(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        stride=1,
        num_point=25,
        block_size=41,
        window_size=120,
    ):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        # self.norm = nn.LayerNorm(out_channels)
        self.norm = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d((stride, 1), (stride, 1))
        self.upfold = UnfoldTemporalWindows(kernel_size, 1, 1)
        conv_init(self.conv)

        self.dropS = DropBlock_Ske(num_point=num_point)
        self.dropT = DropBlockT_1d(block_size=block_size)
        self.attention0s = nn.Parameter(
            torch.ones(1, out_channels, 1, kernel_size) / kernel_size,
            requires_grad=True,
        )

    def forward(self, x, keep_prob, A):
        # x = self.conv(self.pool(x))
        x = self.pool(self.conv(x))
        n, c, t, v = x.shape
        # x = rearrange(x, 'b c t v -> (b t) v c')
        # x = self.norm(x)
        # x = rearrange(x, '(b t) v c -> b c t v', b=n)
        x = self.norm(x)
        upfold = self.upfold(x)
        attention = (
            F.softmax(torch.einsum("nctv,nctuv->nctu", x, upfold) / v, -1)
            + self.attention0s
        )
        # print(attention[0,0,:5,:5])
        x = torch.einsum("nctuv, nctu->nctv", upfold, attention)

        x = self.dropT(self.dropS(x, keep_prob, A), keep_prob)
        return x


class global_tan(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        stride=1,
        num_point=25,
        block_size=41,
        window_size=120,
    ):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, 1),
            padding=(0, 0),
            stride=(1, 1),
        )
        self.norm = nn.LayerNorm(out_channels)
        # self.norm = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d((stride, 1), (stride, 1))
        conv_init(self.conv)
        conv_init(self.conv2)

        self.dropS = DropBlock_Ske(num_point=num_point)
        self.dropT = DropBlockT_1d(block_size=block_size)
        self.attention0s = nn.Parameter(
            torch.zeros(1, out_channels, window_size, window_size)
            + torch.eye(window_size),
            requires_grad=True,
        )

    def forward(self, x, keep_prob, A):
        # x = self.conv(self.pool(x))
        x = self.pool(self.conv(x))
        n, c, t, v = x.shape
        x = rearrange(x, "b c t v -> (b t) v c")
        x = self.norm(x)
        x = rearrange(x, "(b t) v c -> b c t v", b=n)
        # x = self.norm(x)
        x_global = self.conv2(x)
        attention = (
            F.softmax(torch.einsum("nctv,ncyv->ncty", x, x_global) / v, -1)
            + self.attention0s
        )
        # print(attention[0,0,:5,:5])
        x = torch.einsum("ncyv, ncty->nctv", x_global, attention)

        x = self.dropT(self.dropS(x, keep_prob, A), keep_prob)
        return x


class Edge_conv(nn.Module):
    def __init__(self, in_channels, num_joint=27, edge_num=6):
        super().__init__()
        self.num_joint = num_joint
        self.edge_num = edge_num
        self.joint_edge_connections = nn.Parameter(
            torch.randn(num_joint, edge_num), requires_grad=True
        )
        self.edge_joint_connections = nn.Parameter(
            torch.randn(edge_num, num_joint), requires_grad=True
        )
        trunc_normal_(self.joint_edge_connections, std=0.02)
        trunc_normal_(self.edge_joint_connections, std=0.02)
        self.norm1 = nn.LayerNorm(in_channels)
        self.transform = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.alphas = nn.Parameter(torch.ones(1, 1, 1, edge_num), requires_grad=True)
        self.betas = nn.Parameter(torch.ones(1, 1, 1, num_joint), requires_grad=True)
        self.norm2 = nn.LayerNorm(in_channels)
        mlp_ratio = 4
        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=in_channels,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

    def forward(self, x):
        n, c, t, v = x.shape
        x = rearrange(x, "b c t v -> (b t) v c")
        x = self.norm1(x)
        x = rearrange(x, "(b t) v c -> b c t v", b=n)
        x = torch.einsum("nctv, ve->ncte", x, self.joint_edge_connections) * self.alphas
        x = self.transform(x)
        x = torch.einsum("ncte, ev->nctv", x, self.edge_joint_connections) * self.betas

        x_resi = x
        x = rearrange(x, "b c t v -> (b t) v c")
        x = self.norm2(x)
        x = x_resi + rearrange(self.mlp(x), "(b t) v c -> b c t v", b=n)
        return x


class Edge_feature_conv(nn.Module):
    def __init__(self, in_channels, num_joint=27, edge_num=6):
        super().__init__()
        self.num_joint = num_joint
        self.edge_num = edge_num
        self.edge_features = nn.Parameter(
            torch.randn(in_channels, edge_num), requires_grad=True
        )
        trunc_normal_(self.edge_features, std=0.02)
        self.norm1 = nn.LayerNorm(in_channels)
        self.transform = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.alphas = nn.Parameter(torch.ones(1, 1, edge_num, 1), requires_grad=True)
        """self.norm2 = nn.LayerNorm(in_channels)
        mlp_ratio = 4
        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = Mlp(in_features=in_channels, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.0)"""

    def forward(self, x):
        n, c, t, v = x.shape
        x = rearrange(x, "b c t v -> (b t) v c")
        x = self.norm1(x)
        x = rearrange(x, "(b t) v c -> b c t v", b=n)
        x = self.transform(x)
        attention = (
            F.tanh(torch.einsum("nctv, ce->ntev", x, self.edge_features))
            / (c)
            * self.alphas
        )
        x = torch.einsum("ce,ntev->nctv", (self.edge_features, attention))

        """x_resi = x
        x = rearrange(x, 'b c t v -> (b t) v c')
        x = self.norm2(x)
        x = x_resi + rearrange(self.mlp(x), '(b t) v c -> b c t v', b=n)"""
        return x


class Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        groups,
        num_point,
        block_size,
        stride=1,
        residual=True,
        use_grpe=True,
        is_first=False,
        window_size=120,
        **kwargs
    ):

        super().__init__()
        tmp_c = out_channels if is_first else in_channels
        self.san = unit_san(
            in_channels,
            tmp_c,
            A,
            groups,
            num_point,
            is_first=is_first,
            use_grpe=use_grpe,
            **kwargs
        )

        # self.tcn = unit_tcn(tmp_c, out_channels, stride=stride, num_point=num_point)
        # self.tcn2 = unit_tan(tmp_c, out_channels, stride=stride, num_point=num_point, window_size=window_size)
        # self.tcn2 = global_tan(tmp_c, out_channels, stride=stride, num_point=num_point, window_size=window_size)
        # self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)
        # self.tcn = MSTCN(tmp_c, out_channels, stride=stride, num_point=num_point)

        self.tcn = MSTCN(tmp_c, out_channels, stride=1, num_point=num_point)
        self.tcn2 = MSTCN(out_channels, out_channels, stride=1, num_point=num_point)
        self.tcn3 = MSTCN(
            out_channels, out_channels, stride=stride, num_point=num_point
        )
        # self.tcn4 = MSTCN(out_channels, out_channels, stride=stride, num_point=num_point)

        # self.tcn = unit_tcn(tmp_c, out_channels, stride=1, num_point=num_point)
        # self.tcn2 = unit_tan(out_channels, out_channels, stride=stride, num_point=num_point, window_size=window_size)

        self.relu = nn.ReLU()

        self.attention = True
        if self.attention:
            print("Attention Enabled!")
            self.sigmoid = nn.Sigmoid()
            # temporal attention
            self.conv_ta = nn.Conv1d(tmp_c, 1, 9, padding=4)
            nn.init.constant_(self.conv_ta.weight, 0)
            nn.init.constant_(self.conv_ta.bias, 0)
            # s attention
            ker_jpt = num_point - 1 if not num_point % 2 else num_point
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(tmp_c, 1, ker_jpt, padding=pad)
            nn.init.xavier_normal_(self.conv_sa.weight)
            nn.init.constant_(self.conv_sa.bias, 0)
            # channel attention
            rr = 2
            self.fc1c = nn.Linear(tmp_c, tmp_c // rr)
            self.fc2c = nn.Linear(tmp_c // rr, tmp_c)
            nn.init.kaiming_normal_(self.fc1c.weight)
            nn.init.constant_(self.fc1c.bias, 0)
            nn.init.constant_(self.fc2c.weight, 0)
            nn.init.constant_(self.fc2c.bias, 0)

        self.A = nn.Parameter(
            torch.tensor(
                np.sum(
                    np.reshape(A.astype(np.float32), [3, num_point, num_point]), axis=0
                ),
                dtype=torch.float32,
                requires_grad=False,
            ),
            requires_grad=False,
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn_skip(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        self.dropSke = DropBlock_Ske(num_point=num_point)
        self.dropT_skip = DropBlockT_1d(block_size=block_size)

    def forward(self, x, keep_prob):
        y = self.san(x)
        if self.attention:
            # spatial attention
            se = y.mean(-2)  # N C V
            se1 = self.sigmoid(self.conv_sa(se))
            y = y * se1.unsqueeze(-2) + y
            # a1 = se1.unsqueeze(-2)

            # temporal attention
            se = y.mean(-1)
            se1 = self.sigmoid(self.conv_ta(se))
            y = y * se1.unsqueeze(-1) + y
            # a2 = se1.unsqueeze(-1)

            # channel attention
            se = y.mean(-1).mean(-1)
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            # a3 = se2.unsqueeze(-1).unsqueeze(-1)
        # y = self.tcn(y, keep_prob, self.A)
        y = self.tcn(y, keep_prob, self.A)
        y = self.tcn2(y, keep_prob, self.A)
        y = self.tcn3(y, keep_prob, self.A)
        # y = self.tcn4(y, keep_prob, self.A)
        # y = self.tcn(y, keep_prob, self.A) + self.tcn2(y, keep_prob, self.A) * self.weight
        x_skip = self.dropT_skip(
            self.dropSke(self.residual(x), keep_prob, self.A), keep_prob
        )
        return self.relu(y + x_skip)


class MSTCN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=[5, 7],
        stride=1,
        num_point=27,
        block_size=41,
    ):

        super().__init__()
        self.num_branches = len(kernel_sizes)
        assert (
            out_channels % (self.num_branches) == 0
        ), "# out channels should be multiples of # branches"

        # Multiple branches of temporal convolution
        branch_channels = out_channels // self.num_branches

        # Temporal Convolution branches
        self.branches = nn.ModuleList(
            [
                unit_tcn(
                    in_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    num_point=num_point,
                    block_size=block_size,
                )
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x, keep_prob, A):
        # Input dim: (N,C,T,V)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x, keep_prob, A)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        return out


class MultiScale_TemporalConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilations=[1, 2, 3, 4],
        residual=True,
        residual_kernel_size=1,
        num_point=25,
        block_size=41,
    ):

        super().__init__()
        self.num_branches = len(dilations)
        assert (
            out_channels % (self.num_branches) == 0
        ), "# out channels should be multiples of # branches"

        # Multiple branches of temporal convolution
        branch_channels = out_channels // self.num_branches

        # Temporal Convolution branches
        self.branches = nn.ModuleList(
            [
                unit_tcn_dilated(
                    in_channels,
                    branch_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    num_point=num_point,
                    dilation=dilation,
                    block_size=block_size,
                )
                for dilation in dilations
            ]
        )

    def forward(self, x, keep_prob, A):
        # Input dim: (N,C,T,V)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x, keep_prob, A)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        return out


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class unit_tcn_dilated(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=9,
        stride=1,
        dilation=1,
        num_point=25,
        block_size=41,
    ):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

        self.dropS = DropBlock_Ske(num_point=num_point)
        self.dropT = DropBlockT_1d(block_size=block_size)

    def forward(self, x, keep_prob, A):
        x = self.bn(self.conv(x))
        x = self.dropT(self.dropS(x, keep_prob, A), keep_prob)
        return x


class Model(nn.Module):
    def __init__(
        self,
        num_class=60,
        num_point=25,
        num_person=2,
        groups=8,
        block_size=41,
        graph=None,
        graph_args=dict(),
        in_channels=3,
        inner_dim=64,
        drop_layers=3,
        depth=4,
        s_num_heads=1,
        window_size=120,
    ):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.drop_layers = depth - drop_layers

        inner_dim_expansion = [2 ** (i) for i in range(0, depth)]

        self.layers = nn.ModuleList(
            [
                (
                    Block(
                        in_channels,
                        inner_dim,
                        A,
                        groups,
                        num_point,
                        block_size,
                        residual=False,
                        window_size=window_size,
                        i=i,
                        is_first=True,
                    )
                    if i == 0
                    else Block(
                        inner_dim * inner_dim_expansion[i - 1],
                        inner_dim * inner_dim_expansion[i],
                        A,
                        groups,
                        num_point,
                        block_size,
                        stride=inner_dim_expansion[i] // inner_dim_expansion[i - 1],
                        residual=True,
                        window_size=window_size // inner_dim_expansion[i],
                        i=i,
                    )
                )
                for i in range(depth)
            ]
        )

        self.fc = nn.Linear(inner_dim * inner_dim_expansion[-1], num_class)
        nn.init.normal(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)

    def forward(self, x, keep_prob=0.9):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = (
            x.view(N, M, V, C, T)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(N * M, C, T, V)
        )
        for u, blk in enumerate(self.layers):
            x = blk(x, 1.0 if u < self.drop_layers else keep_prob)

        # N*M,C,T,V
        c_new = x.size(1)

        # print(x.size())
        # print(N, M, c_new)

        # x = x.view(N, M, c_new, -1)
        x = x.reshape(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        return self.fc(x)
