"""Taken from https://github.com/L0SG/WaveFlow"""
from torch import nn
import torch
from torch.nn import functional as F
from math import log, pi
from torch.distributions.normal import Normal


class WaveFlowModel(nn.Module):
    def __init__(self, in_channel, cin_channel, res_channel, n_height, n_flow, n_layer, layers_per_dilation_h_cycle, upscaling_fact,
                 bipartize=False):
        super().__init__()
        self.in_channel = in_channel
        self.cin_channel = cin_channel
        self.res_channel = res_channel
        self.n_height = n_height
        self.n_flow = n_flow
        self.n_layer = n_layer

        self.layers_per_dilation_h_cycle = layers_per_dilation_h_cycle
        self.bipartize = bipartize
        if self.bipartize:
            print("INFO: bipartization version for permutation is on for reverse_order. Half the number of flows will use bipartition & reverse over height.")
        self.flows = nn.ModuleList()
        for i in range(self.n_flow):
            self.flows.append(Flow(self.in_channel, self.cin_channel, n_flow=self.n_flow, filter_size=self.res_channel,
                                   num_layer=self.n_layer, num_height=self.n_height,
                                   layers_per_dilation_h_cycle=self.layers_per_dilation_h_cycle,
                                   bipartize=self.bipartize))

        self.upsample_conv = nn.ModuleList()
        for s in [upscaling_fact, upscaling_fact]:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))

        self.upsample_conv_kernel_size = (2*s)**2
        self.upsample_conv_stride = s**2


    def forward(self, x, c, debug=False):
        x = x.unsqueeze(1)
        B, _, T = x.size()
        #  Upsample spectrogram to size of audio
        c = self.upsample(c)
        assert(c.size(2) >= x.size(2))
        if c.size(2) > x.size(2):
            c = c[:, :, :x.size(2)]

        x, c = squeeze_to_2d(x, c, h=self.n_height)
        out = x

        logdet = 0

        if debug:
            list_log_s, list_t  = [], []

        for i, flow in enumerate(self.flows):
            i_flow = i
            out, c, logdet_new, log_s, t = flow(out, c, i_flow, debug)
            if debug:
                list_log_s.append(log_s)
                list_t.append(t)
            logdet = logdet + logdet_new

        if debug:
            return out, logdet, list_log_s, list_t
        else:
            return out, logdet

    def reverse(self, c, temp=1.0, debug_z=None):
        # plain implementation of reverse ops
        c = self.upsample(c)
        # trim conv artifacts. maybe pad spec to kernel multiple
        # time_cutoff = self.upsample_conv_kernel_size - self.upsample_conv_stride
        # c = c[:, :, :-time_cutoff]

        B, _, T_c = c.size()

        _, c = squeeze_to_2d(None, c, h=self.n_height)

        if debug_z is None:
            # sample gaussian noise that matches c
            q_0 = Normal(c.new_zeros((B, 1, c.size()[2], c.size()[3])), c.new_ones((B, 1, c.size()[2], c.size()[3])))
            z = q_0.sample() * temp
        else:
            z = debug_z

        for i, flow in enumerate(self.flows[::-1]):
            i_flow = self.n_flow - (i+1)
            z, c = flow.reverse(z, c, i_flow)

        x = unsqueeze_to_1d(z, self.n_height)

        return x

    def reverse_fast(self, c, temp=1.0, debug_z=None):
        # optimized reverse without redundant computations from conv queue
        c = self.upsample(c)
        # trim conv artifacts. maybe pad spec to kernel multiple
        time_cutoff = self.upsample_conv_kernel_size - self.upsample_conv_stride
        c = c[:, :, :-time_cutoff]

        B, _, T_c = c.size()

        _, c = squeeze_to_2d(None, c, h=self.n_height)

        if debug_z is None:
            # sample gaussian noise that matches c
            q_0 = Normal(c.new_zeros((B, 1, c.size()[2], c.size()[3])), c.new_ones((B, 1, c.size()[2], c.size()[3])))
            z = q_0.sample() * temp
        else:
            z = debug_z

        for i, flow in enumerate(self.flows[::-1]):
            i_flow = self.n_flow - (i+1)
            z, c = flow.reverse_fast(z, c, i_flow)

        x = unsqueeze_to_1d(z, self.n_height)

        return x

    def upsample(self, c):
        c = c.unsqueeze(1)
        for f in self.upsample_conv:
            c = f(c)
        c = c.squeeze(1)
        return c

    def remove_weight_norm(self):
        # remove weight norm from all weights
        for layer in self.upsample_conv.children():
            try:
                torch.nn.utils.remove_weight_norm(layer)
            except ValueError:
                pass
        for flow in self.flows.children():
            net = flow.coupling.net
            torch.nn.utils.remove_weight_norm(net.front_conv[0].conv)
            for resblock in net.res_blocks.children():
                torch.nn.utils.remove_weight_norm(resblock.filter_gate_conv.conv)
                torch.nn.utils.remove_weight_norm(resblock.filter_gate_conv_c)
                torch.nn.utils.remove_weight_norm(resblock.res_skip_conv)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("weight_norm removed: {} params".format(total_params))

    def fuse_conditioning_layers(self):
        # fuse mel-spec conditioning layers into one big conv weight
        for flow in self.flows.children():
            net = flow.coupling.net
            cin_channels = net.res_blocks[0].cin_channels
            out_channels = net.res_blocks[0].out_channels
            fused_filter_gate_conv_c = nn.Conv2d(cin_channels, 2*out_channels*self.n_layer, kernel_size=1)
            fused_filter_gate_conv_c_weight = []
            fused_filter_gate_conv_c_bias = []
            for resblock in net.res_blocks.children():
                fused_filter_gate_conv_c_weight.append(resblock.filter_gate_conv_c.weight)
                fused_filter_gate_conv_c_bias.append(resblock.filter_gate_conv_c.bias)
                del resblock.filter_gate_conv_c

            fused_filter_gate_conv_c.weight = torch.nn.Parameter(torch.cat(fused_filter_gate_conv_c_weight).clone())
            fused_filter_gate_conv_c.bias = torch.nn.Parameter(torch.cat(fused_filter_gate_conv_c_bias).clone())
            flow.coupling.net.fused_filter_gate_conv_c = fused_filter_gate_conv_c

        print("INFO: conditioning layers fused for performance: only reverse_fast function can be used for inference!")
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("model after optimization: {} params".format(total_params))


class WaveFlowCoupling2D(nn.Module):
    def __init__(self, in_channel, cin_channel, filter_size=256, num_layer=6, num_height=None,
                 layers_per_dilation_h_cycle=3):
        super().__init__()
        assert num_height is not None
        self.in_channel = in_channel
        self.num_height = num_height
        self.layers_per_dilation_h_cycle = layers_per_dilation_h_cycle
        # dilation for width & height generation loop
        self.dilation_h = []
        self.dilation_w = []
        self.kernel_size = 3
        for i in range(num_layer):
            self.dilation_h.append(2 ** (i % self.layers_per_dilation_h_cycle))
            self.dilation_w.append(2 ** i)

        self.num_layer = num_layer
        self.filter_size = filter_size
        self.net = Wavenet2D(in_channels=in_channel, out_channels=filter_size,
                             num_layers=num_layer, residual_channels=filter_size,
                             gate_channels=filter_size, skip_channels=filter_size,
                             kernel_size=3, cin_channels=cin_channel, dilation_h=self.dilation_h,
                             dilation_w=self.dilation_w)

        # projector for log_s and t
        self.proj_log_s_t = ZeroConv2d(filter_size, 2*in_channel)

    def forward(self, x, c=None, debug=False):
        x_0, x_in = x[:, :, :1, :], x[:, :, :-1, :]
        c_in = c[:, :, 1:, :]

        feat = self.net(x_in, c_in)
        log_s_t = self.proj_log_s_t(feat)
        log_s = log_s_t[:, :self.in_channel]
        t = log_s_t[:, self.in_channel:]

        x_out = x[:, :, 1:, :]
        x_out, logdet_af = apply_affine_coupling_forward(x_out, log_s, t)

        out = torch.cat((x_0, x_out), dim=2)
        logdet = torch.sum(log_s)

        if debug:
            return out, logdet, log_s, t
        else:
            return out, logdet, None, None

    def reverse(self, z, c=None):
        x = z[:, :, 0:1, :]

        # pre-compute conditioning tensors and cache them
        c_cache = []
        for i, resblock in enumerate(self.net.res_blocks):
            filter_gate_conv_c = resblock.filter_gate_conv_c(c)
            c_cache.append(filter_gate_conv_c)
        c_cache = torch.stack(c_cache)  # [num_layers, batch_size, res_channels, width, height]

        for i_h in range(1, self.num_height):
            feat = self.net.reverse(x, c_cache[:, :, :, 1:i_h+1, :])[:, :, -1, :].unsqueeze(2)
            log_s_t = self.proj_log_s_t(feat)
            log_s = log_s_t[:, :self.in_channel]
            t = log_s_t[:, self.in_channel:]

            x_new = apply_affine_coupling_inverse(z[:, :, i_h, :].unsqueeze(2), log_s, t).unsqueeze(2)
            x = torch.cat((x, x_new), 2)

        return x, c

    def reverse_fast(self, z, c=None):
        x = z[:, :, 0:1, :]
        # initialize conv queue
        self.net.conv_queue_init(x)

        # pre-compute conditioning tensors and cache them
        c_cache =self.net.fused_filter_gate_conv_c(c)
        c_cache = c_cache.reshape(c_cache.shape[0], self.num_layer, self.filter_size*2, c_cache.shape[2], c_cache.shape[3])
        c_cache = c_cache.permute(1, 0, 2, 3, 4) # [num_layers, batch_size, res_channels, height, width]

        x_new = x  # initial first row
        for i_h in range(1, self.num_height):
            feat = self.net.reverse_fast(x_new, c_cache[:, :, :, i_h:i_h+1, :])[:, :, -1, :].unsqueeze(2)

            log_s_t = self.proj_log_s_t(feat)
            log_s = log_s_t[:, :self.in_channel]
            t = log_s_t[:, self.in_channel:]

            x_new = apply_affine_coupling_inverse(z[:, :, i_h, :].unsqueeze(2), log_s, t).unsqueeze(2)
            x = torch.cat((x, x_new), 2)

        return x, c


class Flow(nn.Module):
    def __init__(self, in_channel, cin_channel, n_flow, filter_size, num_layer, num_height, layers_per_dilation_h_cycle, bipartize=False):
        super().__init__()

        self.coupling = WaveFlowCoupling2D(in_channel, cin_channel, filter_size=filter_size, num_layer=num_layer,
                                           num_height=num_height,
                                           layers_per_dilation_h_cycle=layers_per_dilation_h_cycle, )
        self.n_flow = n_flow  # useful for selecting permutation
        self.bipartize = bipartize

    def forward(self, x, c=None, i=None, debug=False):
        logdet = 0

        out, logdet_af, log_s, t = self.coupling(x, c, debug)
        logdet = logdet + logdet_af

        if i < int(self.n_flow / 2):
             # vanilla reverse_order ops
            out = reverse_order(out)
            c = reverse_order(c)
        else:
            if self.bipartize:
                # bipartization & reverse_order ops
                out = bipartize_reverse_order(out)
                c = bipartize_reverse_order(c)
            else:
                out = reverse_order(out)
                c = reverse_order(c)

        if debug:
            return out, c, logdet, log_s, t
        else:
            return out, c, logdet, None, None

    def reverse(self, z, c, i):
        if i < int(self.n_flow / 2):
            z = reverse_order(z)
            c = reverse_order(c)
        else:
            if self.bipartize:
                z = bipartize_reverse_order(z)
                c = bipartize_reverse_order(c)
            else:
                z = reverse_order(z)
                c = reverse_order(c)

        z, c = self.coupling.reverse(z, c)

        return z, c

    def reverse_fast(self, z, c, i):
        if i < int(self.n_flow / 2):
            z = reverse_order(z)
            c = reverse_order(c)
        else:
            if self.bipartize:
                z = bipartize_reverse_order(z)
                c = bipartize_reverse_order(c)
            else:
                z = reverse_order(z)
                c = reverse_order(c)

        z, c = self.coupling.reverse_fast(z, c)

        return z, c


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a+input_b
    t_act = torch.tanh(in_act[:, :n_channels_int])
    s_act = torch.sigmoid(in_act[:, n_channels_int:])
    acts = t_act * s_act
    return acts

@torch.jit.script
def fused_res_skip(tensor, res_skip, n_channels):
    n_channels_int = n_channels[0]
    res = res_skip[:, :n_channels_int]
    skip = res_skip[:, n_channels_int:]
    return (tensor + res), skip

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation_h=1, dilation_w=1,
                 causal=True):
        super(Conv2D, self).__init__()
        self.causal = causal
        self.dilation_h, self.dilation_w = dilation_h, dilation_w

        if self.causal:
            self.padding_h = dilation_h * (kernel_size - 1)  # causal along height
        else:
            self.padding_h = dilation_h * (kernel_size - 1) // 2
        self.padding_w = dilation_w * (kernel_size - 1) // 2  # noncausal along width
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              dilation=(dilation_h, dilation_w), padding=(self.padding_h, self.padding_w))
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, tensor):
        out = self.conv(tensor)
        if self.causal and self.padding_h != 0:
            out = out[:, :, :-self.padding_h, :]
        return out

    def reverse_fast(self, tensor):
        self.conv.padding = (0, self.padding_w)
        out = self.conv(tensor)
        return out


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 1, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)
        return out

class ResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size,
                 cin_channels=None, local_conditioning=True, dilation_h=None, dilation_w=None,
                 causal=True):
        super(ResBlock2D, self).__init__()
        self.out_channels = out_channels
        self.local_conditioning = local_conditioning
        self.cin_channels = cin_channels
        self.skip = True
        assert in_channels == out_channels == skip_channels

        self.filter_gate_conv = Conv2D(in_channels, 2*out_channels, kernel_size, dilation_h, dilation_w, causal=causal)

        self.filter_gate_conv_c = nn.Conv2d(cin_channels, 2*out_channels, kernel_size=1)
        self.filter_gate_conv_c = nn.utils.weight_norm(self.filter_gate_conv_c)
        nn.init.kaiming_normal_(self.filter_gate_conv_c.weight)

        self.res_skip_conv = nn.Conv2d(out_channels, 2*in_channels, kernel_size=1)
        self.res_skip_conv = nn.utils.weight_norm(self.res_skip_conv)
        nn.init.kaiming_normal_(self.res_skip_conv.weight)


    def forward(self, tensor, c=None):
        n_channels_tensor = torch.IntTensor([self.out_channels])

        h_filter_gate = self.filter_gate_conv(tensor)
        c_filter_gate = self.filter_gate_conv_c(c)
        out = fused_add_tanh_sigmoid_multiply(h_filter_gate, c_filter_gate, n_channels_tensor)

        res_skip = self.res_skip_conv(out)

        return fused_res_skip(tensor, res_skip, n_channels_tensor)


    def reverse(self, tensor, c=None):
        # used for reverse. c is a cached tensor
        h_filter_gate = self.filter_gate_conv(tensor)
        n_channels_tensor = torch.IntTensor([self.out_channels])
        out = fused_add_tanh_sigmoid_multiply(h_filter_gate, c, n_channels_tensor)

        res_skip = self.res_skip_conv(out)

        return fused_res_skip(tensor, res_skip, n_channels_tensor)

    def reverse_fast(self, tensor, c=None):
        h_filter_gate = self.filter_gate_conv.reverse_fast(tensor)
        n_channels_tensor = torch.IntTensor([self.out_channels])
        out = fused_add_tanh_sigmoid_multiply(h_filter_gate, c, n_channels_tensor)

        res_skip = self.res_skip_conv(out)

        return fused_res_skip(tensor[:, :, -1:, :], res_skip, n_channels_tensor)


class Wavenet2D(nn.Module):
    # a variant of WaveNet-like arch that operates on 2D feature for WF
    def __init__(self, in_channels=1, out_channels=2, num_layers=6,
                 residual_channels=256, gate_channels=256, skip_channels=256,
                 kernel_size=3, cin_channels=80, dilation_h=None, dilation_w=None,
                 causal=True):
        super(Wavenet2D, self).__init__()
        assert dilation_h is not None and dilation_w is not None

        self.residual_channels = residual_channels
        self.skip = True if skip_channels is not None else False

        self.front_conv = nn.Sequential(
            Conv2D(in_channels, residual_channels, 1, 1, 1, causal=causal),
        )

        self.res_blocks = nn.ModuleList()

        for n in range(num_layers):
            self.res_blocks.append(ResBlock2D(residual_channels, gate_channels, skip_channels, kernel_size,
                                              cin_channels=cin_channels, local_conditioning=True,
                                              dilation_h=dilation_h[n], dilation_w=dilation_w[n],
                                              causal=causal))


    def forward(self, x, c=None):
        h = self.front_conv(x)
        skip = 0
        for i, f in enumerate(self.res_blocks):
            h, s = f(h, c)
            skip += s

        return skip

    def reverse(self, x, c=None):
        # used for reverse op. c is cached tesnor
        h = self.front_conv(x)    # [B, 64, 1, 13264]
        skip = 0
        for i, f in enumerate(self.res_blocks):
            c_i = c[i]
            h, s = f.reverse(h, c_i) # modification: conv_queue + previous layer's output concat , c_i + conv_queue update: conv_queue last element & previous layer's output concat
            skip += s
        return skip

    def reverse_fast(self, x, c=None):
        # input: [B, 64, 1, T]
        # used for reverse op. c is cached tesnor
        h = self.front_conv(x)  # [B, 64, 1, 13264]
        skip = 0
        for i, f in enumerate(self.res_blocks):
            c_i = c[i]
            h_new = torch.cat((self.conv_queues[i], h), dim=2)  # [B, 64, 3, T]
            h, s = f.reverse_fast(h_new, c_i)
            self.conv_queues[i] = h_new[:, :, 1:, :]  # cache the tensor to queue
            skip += s

        return skip

    def conv_queue_init(self, x):
        self.conv_queues = []
        B, _, _, W = x.size()
        for i in range(len(self.res_blocks)):
            conv_queue = torch.zeros((B, self.residual_channels, 2, W), device=x.device)
            if x.type() == 'torch.cuda.HalfTensor':
                conv_queue = conv_queue.half()
            self.conv_queues.append(conv_queue)


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)

def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps

def bipartize(x):
    # bipartize the given tensor along height dimension
    # ex: given [H, W] tensor:
    # [0, 4,      [0, 4,
    #  1, 5,       2, 6,
    #  2, 6,       1, 5,
    #  3, 7,] ==>  3, 7,]
    """
    :param x: tensor with shape [B, 1, H, W]
    :return:  same shape with bipartized formulation
    """
    B, _, H, W = x.size()
    assert H % 2 == 0, "height is not even number, bipartize behavior is undefined."
    x_even = x[:, :, ::2, :]
    x_odd = x[:, :, 1::2, :]
    x_out = torch.cat((x_even, x_odd), dim=2)
    return x_out


def unbipartize(x_even, x_odd):
    # reverse op for bipartize
    assert x_even.size() == x_odd.size()
    B, _, H_half, W = x_even.size()
    merged = torch.empty((B, _, H_half*2, W)).to(x_even.device)
    merged[:, :, ::2, :] = x_even
    merged[:, :, 1::2, :] = x_odd

    return merged


def reverse_order(x, dim=2):
    # reverse order of x and c along height dimension
    x = torch.flip(x, dims=(dim,))
    return x


def bipartize_reverse_order(x, dim=2):
    # permutation stragety (b) from waveflow paper
    # ex: given [H, W] tensor:
    # [0, 4,      [1, 5,
    #  1, 5,       0, 4,
    #  2, 6,       3, 7,
    #  3, 7,] ==>  2, 6,]
    """
    :param x: tensor with shape [B, 1, H, W]
    :return:  same shape with permuted height
    """
    B, _, H, W = x.size()
    assert H % 2 == 0, "height is not even number, bipartize behavior is undefined."
    # unsqueeze to (B, _, 1, H, W), reshape to (B, _, 2, H/2, W), then flip on dim with H/2
    x = x.unsqueeze(dim)
    x = x.view(B, _, 2, int(H/2), W)
    x = x.flip(dims=(dim+1,))
    x = x.view(B, _, -1, W)

    return x

def squeeze_to_2d(x, c, h):
    if x is not None:  # during synthesize phase, we feed x as None
        # squeeze 1D waveform x into 2d matrix given height h
        B, C, T = x.size()
        assert T % h == 0, "cannot make 2D matrix of size {} given h={}".format(T, h)
        x = x.view(B, int(T / h), C * h)
        # permute to make column-major 2D matrix of waveform
        x = x.permute(0, 2, 1)
        # unsqueeze to have tensor shape of [B, 1, H, W]
        x = x.unsqueeze(1)

    # same goes to c, but keeping the 2D mel-spec shape
    B, C, T = c.size()
    c = c.view(B, C, int(T / h), h)
    c = c.permute(0, 1, 3, 2)

    return x, c


def unsqueeze_to_1d(x, h):
    # unsqueeze 2d tensor back to 1d representation
    B, C, H, W = x.size()
    assert H == h, "wrong height given, must match model's n_height {} and given tensor height {}.".format(h, H)
    x = x.permute(0, 1, 3, 2)
    x = x.contiguous().view(B, C, -1)
    x = x.squeeze(1)

    return x


def shift_1d(x):
    # shift tensor on height by one for WaveFlowAR modeling
    x = F.pad(x, (0, 0, 1, 0))
    x = x[:, :, :-1, :]
    return x


def apply_affine_coupling_forward(x, log_s, t):
    out = x * torch.exp(log_s) + t
    logdet = torch.sum(log_s)

    return out, logdet


def apply_affine_coupling_inverse(z, log_s, t):
    return ((z - t) * torch.exp(-log_s)).squeeze(2)