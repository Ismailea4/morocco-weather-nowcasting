import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.attn = WindowAttention(dim, (self.window_size, self.window_size), num_heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = nn.LayerNorm(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W

        x = self.expand(x)
        x = x.view(B, H, W, -1)
        x = x.view(B, H, W, 2, 2, -1).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H*2, W*2, -1)
        x = x.view(B, -1, x.size(-1))
        x = self.norm(x)
        return x

class ViTNowcasting(nn.Module):
    def __init__(self, input_channels=4, embed_dim=96, depths=[2,2,6,2], num_heads=[3,6,12,24],
                 window_size=7, seq_length=3, forecast_horizon=1, img_size=(557, 521)):
        super(ViTNowcasting, self).__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = 4.
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # Patch embedding for each frame
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=4, in_chans=input_channels, embed_dim=embed_dim)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.final_res = (patches_resolution[0] // (2 ** (self.num_layers - 1)), patches_resolution[1] // (2 ** (self.num_layers - 1)))

        # Absolute position embedding
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(0.0)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]  # stochastic depth decay rule

        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList([
                SwinBlock(
                    dim=int(embed_dim * 2 ** i_layer),
                    input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2)
                for i in range(depths[i_layer])])
            if i_layer < self.num_layers - 1:
                downsample = PatchMerging(
                    input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                    dim=int(embed_dim * 2 ** i_layer))
                layer.append(downsample)
            self.layers.append(layer)

        # Temporal fusion
        self.temp_fusion = nn.Conv3d(int(embed_dim * 2 ** (self.num_layers - 1)), int(embed_dim * 2 ** (self.num_layers - 1)), kernel_size=(seq_length, 1, 1))

        # Decoder
        self.decoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1, -1, -1):
            upsample = PatchExpand(
                input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                dim=int(embed_dim * 2 ** i_layer))
            decoder_layer = nn.ModuleList([
                SwinBlock(
                    dim=int(embed_dim * 2 ** i_layer),
                    input_resolution=(patches_resolution[0] // (2 ** i_layer), patches_resolution[1] // (2 ** i_layer)),
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2)
                for i in range(depths[i_layer])])
            decoder_layer.append(upsample)
            self.decoder_layers.append(decoder_layer)

        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))
        self.final_conv = nn.Conv2d(int(embed_dim * 2 ** (self.num_layers - 1)), input_channels, 1)

    def forward(self, x):
        # x: (batch, seq, channels, H, W)
        batch_size, seq_len, _, H, W = x.size()

        # Process each frame through encoder
        encoded_frames = []
        for t in range(seq_len):
            frame = x[:, t]  # (batch, channels, H, W)
            x_enc = self.patch_embed(frame) + self.absolute_pos_embed
            x_enc = self.pos_drop(x_enc)

            skip_connections = []
            for i, layer in enumerate(self.layers):
                for block in layer[:-1] if i < len(self.layers)-1 else layer:
                    x_enc = block(x_enc)
                skip_connections.append(x_enc)
                if i < len(self.layers)-1:
                    x_enc = layer[-1](x_enc)  # downsample

            encoded_frames.append(x_enc)

        # Stack encoded frames: (batch, seq, num_patches, dim)
        encoded = torch.stack(encoded_frames, dim=1)

        # Temporal fusion with Conv3D
        B, S, L, C = encoded.shape
        encoded_reshaped = encoded.view(B, C, S, self.final_res[0], self.final_res[1])
        fused = self.temp_fusion(encoded_reshaped).squeeze(2)  # (B, C, H', W')

        # Decoder
        x_dec = fused.view(B, L, C)
        for i, layer in enumerate(self.decoder_layers):
            for block in layer[:-1]:
                x_dec = block(x_dec)
            if i < len(self.decoder_layers)-1:
                x_dec = layer[-1](x_dec)  # upsample

        x_dec = self.norm(x_dec)
        x_dec = x_dec.view(B, int(np.sqrt(L)), int(np.sqrt(L)), C).permute(0, 3, 1, 2)
        output = self.final_conv(x_dec)

        return output.unsqueeze(1)  # (batch, 1, channels, H, W)