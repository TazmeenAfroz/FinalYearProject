import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


# DCA block
class DCA(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes,  out_planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(in_planes,  out_planes, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_planes)
        self.conv_out = nn.Conv2d(out_planes, out_planes, 1)
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x1, x2, x3):
        g1 = self.bn1(self.conv1(F.adaptive_avg_pool2d(x1, 1)))
        g2 = self.bn2(self.conv2(F.adaptive_avg_pool2d(x2, 1)))
        g3 = self.bn3(self.conv3(F.adaptive_avg_pool2d(x3, 1)))
        g  = torch.relu(g1 + g2 + g3)
        return self.sigmoid(self.conv_out(g))


class BottleneckDCA(Bottleneck):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dca = DCA(self.conv1.out_channels, self.conv3.out_channels)

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)));   x1 = out
        out = self.relu(self.bn2(self.conv2(out)));  x2 = out
        out = self.bn3(self.conv3(out));             x3 = out

        W   = self.dca(x1, x2, x3)
        out = x3 * W

        if self.downsample is not None:
            identity = self.downsample(x)

        return self.relu(out + identity)


# self attention
class SymmetricCrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()

        self.enc_cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dec_cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.enc_self  = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dec_self  = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.enc_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout))
        self.dec_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout))

        self.norm_enc_cross    = nn.LayerNorm(d_model)
        self.norm_enc_self     = nn.LayerNorm(d_model)
        self.norm_enc_ffn      = nn.LayerNorm(d_model)
        self.norm_dec_cross    = nn.LayerNorm(d_model)
        self.norm_dec_cross_im = nn.LayerNorm(d_model)
        self.norm_dec_self     = nn.LayerNorm(d_model)
        self.norm_dec_ffn      = nn.LayerNorm(d_model)

        self.pool_W = nn.Linear(d_model, d_model)
        self.pool_v = nn.Linear(d_model, 1, bias=False)
        self.pool_b = nn.Parameter(torch.zeros(d_model))

    def _one_to_n_fusion(self, f_j, G):
        f_j_n = self.norm_enc_cross(f_j)
        G_n   = self.norm_dec_cross(G)

        G_cross, _  = self.dec_cross(G_n, f_j_n, f_j_n)
        G_fj        = G + G_cross

        G_im_n      = self.norm_dec_cross_im(G_fj)
        f_j_cross, _= self.enc_cross(f_j_n, G_im_n, G_im_n)
        f_j_up      = f_j + f_j_cross

        return f_j_up, G_fj

    def forward(self, enc, dec):
        ne = enc.size(0)

        enc_updated = []
        dec_inter   = []
        for j in range(ne):
            f_j_up, G_fj = self._one_to_n_fusion(enc[j:j+1], dec)
            enc_updated.append(f_j_up)
            dec_inter.append(G_fj)

        enc = torch.cat(enc_updated, dim=0)

        G_c     = torch.stack(dec_inter, dim=0)
        G_c     = G_c.permute(2, 1, 0, 3)
        energy  = torch.tanh(self.pool_W(G_c) + self.pool_b)
        scores  = self.pool_v(energy)
        weights = F.softmax(scores, dim=2)
        dec     = (G_c * weights).sum(dim=2)
        dec     = dec.permute(1, 0, 2)

        enc_n = self.norm_enc_self(enc)
        enc   = enc + self.enc_self(enc_n, enc_n, enc_n)[0]

        dec_n = self.norm_dec_self(dec)
        dec   = dec + self.dec_self(dec_n, dec_n, dec_n)[0]

        enc = enc + self.enc_ffn(self.norm_enc_ffn(enc))
        dec = dec + self.dec_ffn(self.norm_dec_ffn(dec))

        return enc, dec


class GazeSymCAT(nn.Module):
    def __init__(self, d_model=512, num_blocks=2, use_head_pose=False):
        super().__init__()
        self.use_head_pose = use_head_pose
        self.d_model       = d_model

        self.backbone = ResNet(block=BottleneckDCA, layers=[3, 4, 6, 3])
        self._load_pretrained_backbone()
        self.backbone.fc      = nn.Identity()
        self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.proj    = nn.Linear(2048, d_model)
        self.proj_ln = nn.LayerNorm(d_model)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

        self.encoder_pos_embed = nn.Parameter(torch.randn(3, 1, d_model) * 0.02)

        self.query_embed = nn.Parameter(torch.randn(3, 1, d_model) * 0.02)

        if use_head_pose:
            self.head_query_base = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.head_query_proj = nn.Linear(2, d_model)
            nn.init.normal_(self.head_query_proj.weight, std=0.02)
            nn.init.zeros_(self.head_query_proj.bias)

        self.sym_blocks = nn.ModuleList([
            SymmetricCrossAttentionBlock(d_model, nhead=8, dim_feedforward=2048)
            for _ in range(num_blocks)
        ])

        self.face_fc = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(0.2))
        self.eyes_fc = nn.Sequential(
            nn.Linear(d_model * 2, 256), nn.ReLU(), nn.Dropout(0.2))

        if use_head_pose:
            self.head_fc   = nn.Sequential(
                nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(0.1))
            gaze_in_dim = 256 + 256 + 128
        else:
            gaze_in_dim = 256 + 256

        self.gaze_out = nn.Linear(gaze_in_dim, 2)
        nn.init.normal_(self.gaze_out.weight, std=0.01)
        nn.init.zeros_(self.gaze_out.bias)

    def _load_pretrained_backbone(self):
        try:
            state = load_state_dict_from_url(
                'https://download.pytorch.org/models/resnet50-0676ba61.pth',
                progress=False)
            curr       = self.backbone.state_dict()
            compatible = {k: v for k, v in state.items()
                          if k in curr and v.shape == curr[k].shape}
            self.backbone.load_state_dict(compatible, strict=False)
            print(f'[Backbone] Loaded {len(compatible)}/{len(curr)} pretrained weights '
                  f'(DCA params are new — trained from scratch)')
        except Exception as e:
            print(f'[Backbone] Pretrained load failed ({e}) — training from scratch')

    def _extract_feature(self, x):
        feat = self.backbone(x).flatten(1)
        feat = self.proj_ln(self.proj(feat))
        return feat

    def forward(self, face, leye, reye, head_pose=None):
        B = face.size(0)

        f_face = self._extract_feature(face)
        f_leye = self._extract_feature(leye)
        f_reye = self._extract_feature(reye)

        enc = torch.stack([f_face, f_leye, f_reye], dim=0)
        enc = enc + self.encoder_pos_embed.expand(-1, B, -1)

        dec = self.query_embed.expand(-1, B, -1).contiguous()

        if self.use_head_pose and head_pose is not None:
            hp_proj  = self.head_query_proj(head_pose)
            hp_query = (self.head_query_base +
                        hp_proj.unsqueeze(0))
            dec = torch.cat([dec, hp_query], dim=0)

        for block in self.sym_blocks:
            enc, dec = block(enc, dec)

        dec = dec.permute(1, 0, 2)
        g_face = dec[:, 0, :]
        g_leye = dec[:, 1, :]
        g_reye = dec[:, 2, :]

        face_feat = self.face_fc(g_face)
        eyes_feat = self.eyes_fc(torch.cat([g_leye, g_reye], dim=1))

        if self.use_head_pose and head_pose is not None:
            g_hpose   = dec[:, 3, :]
            head_feat = self.head_fc(g_hpose)
            combined  = torch.cat([face_feat, eyes_feat, head_feat], dim=1)
        else:
            combined = torch.cat([face_feat, eyes_feat], dim=1)

        gaze = self.gaze_out(combined)
        return gaze
