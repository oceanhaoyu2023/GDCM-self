"""Neural-network building blocks for the GDCM-EOF chlorophyll-a model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter

class ConvLSTMCell(nn.Module):
 
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
 
        super(ConvLSTMCell, self).__init__()
 
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
 
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2 # 保证在传递过程中 （h,w）不变
        self.bias = bias
 
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim, # i门，f门，o门，g门放在一起计算，然后在split开
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
 
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state # 每个timestamp包含两个状态张量：h和c
 
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis # 把输入张量与h状态张量沿通道维度串联
 
        combined_conv = self.conv(combined) # i门，f门，o门，g门放在一起计算，然后在split开
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
 
        c_next = f * c_cur + i * g  # c状态张量更新
        h_next = o * torch.tanh(c_next) # h状态张量更新
 
        return h_next, c_next # 输出当前timestamp的两个状态张量
 
    def init_hidden(self, batch_size, image_size):
        """
        初始状态张量初始化.第一个timestamp的状态张量0初始化
        :param batch_size:
        :param image_size:
        :return:
        """
        height, width = image_size
        init_h = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        init_c = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        return (init_h,init_c)


class ConvLSTM(nn.Module):
 
    """
    Parameters:参数介绍
        input_dim: Number of channels in input# 输入张量的通道数
        hidden_dim: Number of hidden channels # h,c两个状态张量的通道数，可以是一个列表
        kernel_size: Size of kernel in convolutions # 卷积核的尺寸，默认所有层的卷积核尺寸都是一样的,也可以设定不通lstm层的卷积核尺寸不同
        num_layers: Number of LSTM layers stacked on each other # 卷积层的层数，需要与len(hidden_dim)相等
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers # 是否返回所有lstm层的h状态
        Note: Will do same padding. # 相同的卷积核尺寸，相同的padding尺寸
    Input:输入介绍
        A tensor of size [B, T, C, H, W] or [T, B, C, H, W]# 需要是5维的
    Output:输出介绍
        返回的是两个列表：layer_output_list，last_state_list
        列表0：layer_output_list--单层列表，每个元素表示一层LSTM层的输出h状态,每个元素的size=[B,T,hidden_dim,H,W]
        列表1：last_state_list--双层列表，每个元素是一个二元列表[h,c],表示每一层的最后一个timestamp的输出状态[h,c],h.size=c.size = [B,hidden_dim,H,W]
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:使用示例
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """
 
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()
 
        self._check_kernel_size_consistency(kernel_size)
 
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers) # 转为列表
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers) # 转为列表
        if not len(kernel_size) == len(hidden_dim) == num_layers: # 判断一致性
            raise ValueError('Inconsistent list length.')
 
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
 
        cell_list = []
        for i in range(0, self.num_layers): # 多层LSTM设置
            # 当前LSTM层的输入维度
            # if i==0:
            #     cur_input_dim = self.input_dim
            # else:
            #     cur_input_dim = self.hidden_dim[i - 1]
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1] # 与上等价
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
 
        self.cell_list = nn.ModuleList(cell_list) # 把定义的多个LSTM层串联成网络模型
 
    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: 5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
 
        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            b, _, _, h, w = input_tensor.size()  # 自动获取 b,h,w信息
            hidden_state = self._init_hidden(batch_size=b,image_size=(h, w))
 
        layer_output_list = []
        last_state_list = []
 
        seq_len = input_tensor.size(1) # 根据输入张量获取lstm的长度
        cur_layer_input = input_tensor
 
        for layer_idx in range(self.num_layers): # 逐层计算
 
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len): # 逐个stamp计算
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],cur_state=[h, c])
                output_inner.append(h) # 第 layer_idx 层的第t个stamp的输出状态
 
            layer_output = torch.stack(output_inner, dim=1) # 第 layer_idx 层的第所有stamp的输出状态串联
            cur_layer_input = layer_output # 准备第layer_idx+1层的输入张量
 
            layer_output_list.append(layer_output) # 当前层的所有timestamp的h状态的串联
            last_state_list.append([h, c]) # 当前层的最后一个stamp的输出状态的[h,c]
 
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
 
        return layer_output_list, last_state_list
 
    def _init_hidden(self, batch_size, image_size):
        """
        所有lstm层的第一个timestamp的输入状态0初始化
        :param batch_size:
        :param image_size:
        :return:
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states
 
    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """
        检测输入的kernel_size是否符合要求，要求kernel_size的格式是list或tuple
        :param kernel_size:
        :return:
        """
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')
 
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        扩展到多层lstm情况
        :param param:
        :param num_layers:
        :return:
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


def depth_to_space(tensor, scale_factor):
    num, ch, height, width = tensor.shape
    if ch % (scale_factor * scale_factor) != 0:
        raise ValueError('channel of tensor must be divisible by '
                         '(scale_factor * scale_factor).必须是整数倍.')

    new_ch = ch // (scale_factor * scale_factor)
    new_height = height * scale_factor
    new_width = width * scale_factor

    tensor = tensor.reshape(
        [num, new_ch, scale_factor, scale_factor, height, width])
    tensor = tensor.permute([0, 1, 4, 2, 5, 3])
    tensor = tensor.reshape([num, new_ch, new_height, new_width])
    return tensor


def space_to_depth(tensor, scale_factor):
    num, ch, height, width = tensor.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ValueError('height and widht of tensor must be divisible by '
                         'scale_factor.')

    new_ch = ch * (scale_factor * scale_factor)
    new_height = height // scale_factor
    new_width = width // scale_factor

    tensor = tensor.reshape(
        [num, ch, new_height, scale_factor, new_width, scale_factor])
    # new axis: [num, ch, scale_factor, scale_factor, new_height, new_width]
    tensor = tensor.permute([0, 1, 3, 5, 2, 4])
    tensor = tensor.reshape([num, scale_factor*scale_factor,ch, new_height, new_width])
    return tensor


class GBlockUp(nn.Module): #尺度上升模块 三层
    def __init__(self, in_channels, out_channels):
        super(GBlockUp, self).__init__()
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.Tanh()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3_1 = nn.Conv2d(in_channels, in_channels, 9, stride=1, padding=4)  # 图像大小保持不变 通道数不变
        self.conv3_2 = nn.Conv2d(in_channels, out_channels, 9, stride=1, padding=4)  # 图像大小保持不变 通道数翻倍

    def forward(self, x):
        x1 = self.upsample(x)
        x1 = self.conv1(x1)

        x2 = self.BN(x)
        x2 = self.relu(x2)
        x2 = self.upsample(x2)
        x2 = self.conv3_1(x2)
        x2 = self.BN(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)

        out = x1 + x2

        return out


class GBlock(nn.Module): #尺度上升模块 两层
    def __init__(self, in_channels, out_channels):
        super(GBlock, self).__init__()
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.Tanh()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv3_1 = nn.Conv2d(in_channels, out_channels, 9, stride=1, padding=4)  # 图像大小保持不变 通道数不变

    def forward(self, x):
        # x1 = self.up  sample(x)
        x1 = self.conv1(x)

        x2 = self.BN(x)
        x2 = self.relu(x2)
        # x2 = self.upsample(x2)
        x2 = self.conv3_1(x2)
        x2 = self.BN(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_1(x2)

        out = x1 + x2
#         print()
        return out


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class ConvGRUCell(nn.Module):
    """
    Generate a convolutional GRU cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, activation=torch.sigmoid):

        super().__init__()
        padding = kernel_size//2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate  = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding,stride=1)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding,stride=1)
        self.out_gate    = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding,stride=1)
        self.activation = activation

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)


    def forward(self, x, prev_state=None):

        if prev_state is None:

            # get batch and spatial sizes
            batch_size = x.data.size()[0]
            spatial_size = x.data.size()[2:]

            # generate empty prev_state, if None is provided
            state_size = [batch_size, self.hidden_size] + list(spatial_size)

            if torch.cuda.is_available():
                prev_state = torch.zeros(state_size)

            else:
                prev_state = torch.zeros(state_size)

        combined_1 = torch.cat([x, prev_state], dim=1)
        update = self.activation(self.update_gate(combined_1))
        reset = self.activation(self.reset_gate(combined_1))
        out_inputs = torch.tanh(self.out_gate(torch.cat([x, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


class SequenceGRU(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.conv =SpectralNorm(nn.Conv2d(input_size, input_size, kernel_size=9, padding=4, stride=1))
        self.GBlock = GBlock(input_size,input_size)
        self.GBlockUp = GBlockUp(input_size,input_size)
    def forward(self,x):

        x=self.conv(x)
        x=self.GBlock(x)
        out=self.GBlockUp(x)
        return out


class ConvGRU(nn.Module):

    def __init__(self,input_dim, hidden_dim, kernel_sizes, num_layers,gb_hidden_size):
        """
        Generates a multi-layer convolutional GRU.
        :param input_size: integer. depth dimension of input tensors.
        :param hidden_sizes: integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        :param kernel_sizes: integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        :param n_layers: integer. number of chained `ConvGRUCell`.
        """

        super().__init__()

        self.input_size = input_dim
        self.input_dim =input_dim

        if type(hidden_dim) != list:
            self.hidden_sizes = [hidden_dim]*num_layers
        else:
            assert len(hidden_dim) == num_layers, '`hidden_sizes` must have the same length as n_layers'
            self.hidden_sizes = hidden_dim
        if type(kernel_sizes) != list:
            self.kernel_sizes = [kernel_sizes]*num_layers
        else:
            assert len(kernel_sizes) == num_layers, '`kernel_sizes` must have the same length as n_layers'
            self.kernel_sizes = kernel_sizes

        self.n_layers = num_layers

        cells = nn.ModuleList()
        squenceCells=nn.ModuleList()

        for i in range(self.n_layers):

            if i == 0:
                input_dim = self.input_size
            else:
                input_dim = self.hidden_sizes[i-1]

            cell = ConvGRUCell(self.input_dim[i], self.hidden_sizes[i], 3)

            cells.append(cell)

        self.cells = cells


        for i in range(self.n_layers):

            squenceCell = SequenceGRU(gb_hidden_size[i])

            squenceCells.append(squenceCell)

        self.squenceCells = squenceCells


    def forward(self, x, hidden):
        '''
        Parameters
        ----------
        x : 4D input tensor. (batch, channels, height, width).
        hidden : list of 4D hidden state representations. (batch, channels, height, width).
        Returns
        -------
        upd_hidden : 5D hidden representation. (layer, batch, channels, height, width).
        '''

        input = x
        output = []


        layer_output_list = []
        last_state_list = []
        seq_len = input.size(1)
        cur_layer_input = input

        for layer_idx in range(self.n_layers):
            output_inner=[]
            for t in range(seq_len):


               cell= self.cells[layer_idx]
               cell_hidden = hidden[layer_idx]
               squenceCell=self.squenceCells[layer_idx]

               # pass through layer

               a=cur_layer_input[:, t, :, :, :]

               upd_cell_hidden = cell(cur_layer_input[:, t, :, :, :], cell_hidden) # TODO comment
               upd_cell_hidden=squenceCell(upd_cell_hidden)

               output_inner.append(upd_cell_hidden)

            layer_output = torch.stack(output_inner, dim=1)#每一层layer的所有18个hidden状态输出
            cur_layer_input=layer_output

            layer_output_list.append(layer_output)#所有layer层的18个hidden输出
            last_state_list.append(cell_hidden)    #最后一层的18的hidden输出

        layer_output_list = layer_output_list[-1:]
        last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list


class DBlockDown(nn.Module):  # 下采样
    def __init__(self, in_channels, out_channels):
        super(DBlockDown, self).__init__()
        self.relu = nn.Tanh()
        self.conv1 =SpectralNorm( nn.Conv2d(in_channels, out_channels, 1))
        self.conv3_1 = SpectralNorm(nn.Conv2d(in_channels, in_channels, 9,stride=1,padding=4))#图像大小保持不变
        self.conv3_2 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 9, stride=1, padding=4)) # 图像大小保持不变
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool(x1)

        x2 = self.relu(x)
        x2 = self.conv3_1(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool(x2)
        out = x1 + x2
        return out


class DBlockDownFirst(nn.Module):  # 下采样
    def __init__(self, in_channels, out_channels):
        super(DBlockDownFirst, self).__init__()
        self.relu = nn.Tanh()
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 1))
        self.conv3_1 = SpectralNorm(nn.Conv2d(in_channels, in_channels, 9,stride=1,padding=4))#图像大小保持不变
        self.conv3_2 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 9, stride=1, padding=4)) # 图像大小保持不变
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool(x1)


        x2 = self.conv3_1(x)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool(x2)
        out = x1 + x2
        return out


class DBlock(nn.Module):  # 卷积核，padding,stride待确定
    def __init__(self, in_channels, out_channels):
        super(DBlock, self).__init__()
        self.relu = nn.Tanh()
        self.conv1 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 1))
        self.conv3 = SpectralNorm(nn.Conv2d(in_channels, out_channels, 9,stride=1,padding=4))#图像大小保持不变)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.relu(x)
        x2 = self.conv3(x2)
        x2 = self.relu(x2)
        x2 = self.conv3(x2)
        out = x1 + x2
        return out


class DBlock3D_1(nn.Module):  # 卷积核，padding,stride待确定
    def __init__(self, in_channels, out_channels):
        super(DBlock3D_1, self).__init__()
        self.relu = nn.Tanh()
        self.conv1 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)))
        self.conv3_1 = SpectralNorm(nn.Conv3d(in_channels, in_channels, kernel_size=(7, 7, 7), padding=3,stride=1))
        self.conv3_2 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(7, 7, 7), padding=3, stride=1))
        self.maxpool_3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.maxpool_3d(x1)


        x2 = self.conv3_1(x)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool_3d(x2)
        out = x1 + x2

        return out


class DBlock3D_2(nn.Module):  # 卷积核，padding,stride待确定
    def __init__(self, in_channels, out_channels):
        super(DBlock3D_2, self).__init__()
        self.relu = nn.Tanh()
        self.conv1 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)))
        self.conv3_1 = SpectralNorm(nn.Conv3d(in_channels, in_channels, kernel_size=(7, 7, 7), padding=3,stride=1))
        self.conv3_2 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(7, 7, 7), padding=3, stride=1))
        self.maxpool_3d = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.maxpool_3d(x1)

        x2 = self.relu(x)
        x2 = self.conv3_1(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool_3d(x2)
        out = x1 + x2

        return out


class DBlock3D_2_spatial(nn.Module):  # 卷积核，padding,stride待确定
    def __init__(self, in_channels, out_channels):
        super(DBlock3D_2_spatial, self).__init__()
        self.relu = nn.Tanh()
        self.conv1 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)))
        self.conv3_1 = SpectralNorm(nn.Conv3d(in_channels, in_channels, kernel_size=(7, 7, 7), padding=3,stride=1))
        self.conv3_2 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(7, 7, 7), padding=3, stride=1))
        self.maxpool_3d = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.maxpool_3d(x1)

        x2 = self.relu(x)
        x2 = self.conv3_1(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
        x2 = self.maxpool_3d(x2)
        out = x1 + x2

        return out


class DBlock3D_2_spatial_1(nn.Module):  # 卷积核，padding,stride待确定
    def __init__(self, in_channels, out_channels):
        super(DBlock3D_2_spatial_1, self).__init__()
        self.relu = nn.Tanh()
        self.conv1 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1)))
        self.conv3_1 = SpectralNorm(nn.Conv3d(in_channels, in_channels, kernel_size=(7, 7, 7), padding=3,stride=1))
        self.conv3_2 = SpectralNorm(nn.Conv3d(in_channels, out_channels, kernel_size=(7, 7, 7), padding=3, stride=1))
#         self.maxpool_3d = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):

        x1 = self.conv1(x)
#         x1 = self.maxpool_3d(x1)

        x2 = self.relu(x)
        x2 = self.conv3_1(x2)
        x2 = self.relu(x2)
        x2 = self.conv3_2(x2)
#         x2 = self.maxpool_3d(x2)
        out = x1 + x2

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 9, stride=1,padding=4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        mask=self.sigmoid(x)
        return mask


class LBlock(nn.Module):  # 卷积核，padding,stride待确定
    def __init__(self, in_channels, out_channels):
        super(LBlock, self).__init__()
        self.relu = nn.Tanh()
        self.conv1 = nn.Conv2d(in_channels,out_channels-in_channels, 1)
        self.conv3_1 = nn.Conv2d(in_channels, in_channels, 9, stride=1, padding=4)  # 图像大小保持不变
        self.conv3_2 = nn.Conv2d(in_channels, out_channels, 9, stride=1, padding=4)  # 图像大小保持不变

    def forward(self, x):
        x1 = self.relu(x)
        x1 = self.conv3_1(x1)
        x1 = self.relu(x1)
        x1 = self.conv3_2(x1)

        x2 = self.conv1(x)
        x3=x
        x23=torch.cat([x2,x3],axis=1)

        out = x1 + x23
        return out


class LCStack(nn.Module): #空间模块
    def __init__(self,):
        super(LCStack, self).__init__()
        self.conv3_1 = nn.Conv2d(8, 8, 9, stride=1, padding=4)  # 图像大小保持不变 通道数不变
        self.LBlock_1 = LBlock(8, 24)
        self.LBlock_2 = LBlock(24, 48)
        self.LBlock_3 = LBlock(48, 192)
        self.LBlock_4= LBlock(192, 768)
        self.mask = SpatialAttention()
    def forward(self,x):
        x = self.conv3_1(x)
        x = self.LBlock_1(x)
        x = self.LBlock_2(x)
        x = self.LBlock_3(x)
        mask = self.mask(x)

        x=x*mask
        out = self.LBlock_4(x)

        return out


class FrameStack(nn.Module):
    def __init__(self, inchannels):
        super(FrameStack, self).__init__()
        # self.s2d = space_to_depth()
        self.inchannels = inchannels
        self.DBlockDown_1 = DBlockDown(1,48)
        self.DBlockDown_2 = DBlockDown(48,96)
        self.DBlockDown_3 = DBlockDown(96,192)
        self.DBlockDown_4 = DBlockDown(192,384)
        self.mask = SpatialAttention()
#         self.conv3_1 = SpectralNorm(nn.Conv2d(48,96, 3,stride=1,padding=1))#图像大小保持不变
#         self.conv3_2 = SpectralNorm(nn.Conv2d(96,192, 3, stride=1, padding=1))  # 图像大小保持不变
#         self.conv3_3 = SpectralNorm(nn.Conv2d(192,384 3, stride=1, padding=1))  # 图像大小保持不变
        self.conv3_4 = SpectralNorm(nn.Conv2d(384,768, 9, stride=1, padding=4))  # 图像大小保持不变

    def forward(self, x):
        x_new = x
        x_down_1 = self.DBlockDown_1(x_new)
#         print(x_down_1.shape)
        x_down_2 = self.DBlockDown_2(x_down_1)
        x_down_3 = self.DBlockDown_3(x_down_2)
        x_down_4 = self.DBlockDown_4(x_down_3)
        mask = self.mask(x_down_4)
        x_down_4=x_down_4*mask
        x_end = self.conv3_4(x_down_4)

        return x_end


class conditioningStack(nn.Module):
    def __init__(self, in_channels):
        super(conditioningStack, self).__init__()
        # self.s2d = space_to_depth()
        self.inchannels = inchannels
        self.DBlockDown_1 = DBlockDown(4,24)
        self.DBlockDown_2 = DBlockDown(24,48)
        self.DBlockDown_3 = DBlockDown(48,96)
        self.DBlockDown_4 = DBlockDown(96,192)
        self.relu = nn.Tanh()
        self.conv3_1 = SpectralNorm(nn.Conv2d(24*self.inchannels, 48, 9,stride=1,padding=4))#图像大小保持不变
        self.conv3_2 = SpectralNorm(nn.Conv2d(48*self.inchannels, 96, 9, stride=1, padding=4))  # 图像大小保持不变
        self.conv3_3 = SpectralNorm(nn.Conv2d(96*self.inchannels, 192, 9, stride=1, padding=4))  # 图像大小保持不变
        self.conv3_4 = SpectralNorm(nn.Conv2d(192*self.inchannels, 384, 9, stride=1, padding=4))  # 图像大小保持不变

    def forward(self, x):

        dataList=[]

        for i in range(x.shape[1]):
            x_new=x[:,i,:,:,:]
            x_new=space_to_depth(x_new,2)
            x_new = torch.squeeze(x_new)
            x_new=self.DBlockDown_1(x_new)

            if i==0:
                data_0=x_new
            else:
                data_0=torch.cat((data_0,x_new),1)
                if i == self.inchannels-1:
                  data_0=self.conv3_1(data_0)
                  dataList.append(data_0)

            x_new=self.DBlockDown_2(x_new)

            if i==0:
                data1=x_new
            else:
                data1=torch.cat((data1,x_new),1)
                if i == self.inchannels:
                    # data1 = nn.utils.spectral_norm(data1)
                    data1 = self.conv3_2(data1)
                    dataList.append(data1)
            x_new=self.DBlockDown_3(x_new)

            if i==0:
                data2=x_new
            else:
                data2=torch.cat((data2,x_new),1)
                if i == self.inchannels:
                    # data2 = nn.utils.spectral_norm(data2)
                    data2 = self.conv3_3(data2)
                    dataList.append(data2)
            x_new=self.DBlockDown_4(x_new)

            if i==0:
                data3=x_new
            else:
                data3=torch.cat((data3,x_new),1)
                if i == self.inchannels:
                    # data3 = nn.utils.spectral_norm(data3)
                    data3= self.conv3_4(data3)
                    dataList.append(data3)


        return dataList


class TimeStack(nn.Module):
    def __init__(self, inchannels):
        super(TimeStack, self).__init__()
        # self.s2d = space_to_depth()
        self.inchannels = inchannels
        
        self.DBlockDown3D_1 =DBlock3D_2_spatial_1(1, 24)
        self.DBlockDown3D_2 = DBlock3D_2_spatial(24, 48)
        self.DBlockDown3D_3 =DBlock3D_2_spatial(48, 96)
        self.DBlockDown3D_4 =DBlock3D_2_spatial(96, 192)
        self.Ada = nn.AdaptiveAvgPool3d([1, None, None])

        self.relu = nn.Tanh()
        self.conv3_1 = SpectralNorm(nn.Conv3d(24, 48, kernel_size=(7, 7, 7), padding=(3, 3, 3)))#图像大小保持不变
        self.conv3_2 = SpectralNorm(nn.Conv3d(48, 96, kernel_size=(7, 7, 7), padding=(3, 3, 3)))  # 图像大小保持不变
        self.conv3_3 = SpectralNorm(nn.Conv3d(96, 192, kernel_size=(7, 7, 7), padding=(3, 3, 3)))  # 图像大小保持不变
        self.conv3_4 = SpectralNorm(nn.Conv3d(192, 384, kernel_size=(7, 7, 7), padding=(3, 3, 3)))  # 图像大小保持不变
        
        self.dot_shape_1 = [48, 48]
        self.dot_w_1 = nn.init.normal_(torch.empty(self.dot_shape_1), mean=0.0, std=1.0)
        self.dot_w_1 = nn.Parameter(self.dot_w_1, requires_grad=True)
        
        self.dot_shape_2 = [48, 96]
        self.dot_w_2 = nn.init.normal_(torch.empty(self.dot_shape_2), mean=0.0, std=1.0)
        self.dot_w_2 = nn.Parameter(self.dot_w_2, requires_grad=True)
        
        self.dot_shape_3 = [48, 192]
        self.dot_w_3 = nn.init.normal_(torch.empty(self.dot_shape_3), mean=0.0, std=1.0)
        self.dot_w_3 = nn.Parameter(self.dot_w_3, requires_grad=True)
        
#         self.dot_shape_4 = [48, 384]
#         self.dot_w_4 = nn.init.normal_(torch.empty(self.dot_shape_4), mean=0.0, std=1.0)
#         self.dot_w_4 = nn.Parameter(self.dot_w_4, requires_grad=True)
        
    def forward(self, x):
        dataList = []
        x_new=x[:,:,1:,:]-x[:,:,0:-1,:]
#         x[:,:,1:,:]-x[:,:,0:-1,:]
#         space_to_depth(x,2)
        data_1 = self.DBlockDown3D_1(x_new)
        data_1_1 = self.conv3_1(data_1)
        data_1_1 = self.Ada(data_1_1)
        data1_1_1_query = torch.squeeze(data_1_1, dim=2) # make (N, C, H, W)
        query_c, query_h, query_w = data1_1_1_query.size()[1],data1_1_1_query.size()[2], data1_1_1_query.size()[3]
        data1_1_1_query = data1_1_1_query.permute(0, 2, 3, 1) # make (N, H, W, C)
        data1_1_1_query = torch.reshape(data1_1_1_query, (-1, query_c)) # make (N*H*W, C)
        query_norm_1 = F.normalize(data1_1_1_query, dim=1)
        dot_norm_1 = F.normalize(self.dot_w_1, dim=1)
        s = torch.mm(query_norm_1, dot_norm_1.transpose(0, 1))
        dot_s_vec_1 = F.softmax(s, dim=1)
        data_1_1_feature = torch.mm(dot_s_vec_1, self.dot_w_1)
        data_1_1_feature = torch.reshape(data_1_1_feature, (-1, query_h, query_w, query_c)) # make (N, H, W, C)
        data_1_1_feature = data_1_1_feature.permute(0, 3, 1, 2)
#         print(data_1_1_feature.shape)
        dataList.append(data_1_1_feature)
        
        data_2 = self.DBlockDown3D_2(data_1)
        data_2_2 = self.conv3_2(data_2)
        data_2_2 = self.Ada(data_2_2)
        
        data1_2_2_query = torch.squeeze(data_2_2, dim=2) # make (N, C, H, W)
#         print(data1_2_2_query.size())
        query_c, query_h, query_w = data1_2_2_query.size()[1],data1_2_2_query.size()[2], data1_2_2_query.size()[3]
        data1_2_2_query = data1_2_2_query.permute(0, 2, 3, 1) # make (N, H, W, C)
        data1_2_2_query = torch.reshape(data1_2_2_query, (-1, query_c)) # make (N*H*W, C)
        query_norm_2 = F.normalize(data1_2_2_query, dim=1)
        dot_norm_2 = F.normalize(self.dot_w_2, dim=1)
        s = torch.mm(query_norm_2, dot_norm_2.transpose(0, 1))
        dot_s_vec_2 = F.softmax(s, dim=1)
        data_2_2_feature = torch.mm(dot_s_vec_2, self.dot_w_2)
        data_2_2_feature = torch.reshape(data_2_2_feature, (-1, query_h, query_w, query_c)) # make (N, H, W, C)
        data_2_2_feature = data_2_2_feature.permute(0, 3, 1, 2)
#         print(data_2_2_feature.shape)
        dataList.append(data_2_2_feature)
        
#         dataList.append(data_2_1)
        data_3 = self.DBlockDown3D_3(data_2)
        data_3_3 = self.conv3_3(data_3)
        data_3_3 = self.Ada(data_3_3)
        data1_3_3_query = torch.squeeze(data_3_3, dim=2) # make (N, C, H, W)
        query_c, query_h, query_w = data1_3_3_query.size()[1],data1_3_3_query.size()[2], data1_3_3_query.size()[3]
        data1_3_3_query = data1_3_3_query.permute(0, 2, 3, 1) # make (N, H, W, C)
        data1_3_3_query = torch.reshape(data1_3_3_query, (-1, query_c)) # make (N*H*W, C)
        query_norm_3 = F.normalize(data1_3_3_query, dim=1)
        dot_norm_3 = F.normalize(self.dot_w_3, dim=1)
        s = torch.mm(query_norm_3, dot_norm_3.transpose(0, 1))
        dot_s_vec_3 = F.softmax(s, dim=1)
        data_3_3_feature = torch.mm(dot_s_vec_3, self.dot_w_3)
        data_3_3_feature = torch.reshape(data_3_3_feature, (-1, query_h, query_w, query_c)) # make (N, H, W, C)
        data_3_3_feature = data_3_3_feature.permute(0, 3, 1, 2)
#         print(data_3_3_feature.shape)
        dataList.append(data_3_3_feature)

#         data_4 = self.DBlockDown3D_4(data_3)
#         data_4_4 = self.conv3_4(data_4)
#         data_4_4 = self.Ada(data_4_4)
#         data1_4_4_query = torch.squeeze(data_4_4, dim=2) # make (N, C, H, W)
#         query_c, query_h, query_w = data1_4_4_query.size()[1],data1_4_4_query.size()[2], data1_4_4_query.size()[3]
#         data1_4_4_query = data1_4_4_query.permute(0, 2, 3, 1) # make (N, H, W, C)
#         data1_4_4_query = torch.reshape(data1_4_4_query, (-1, query_c)) # make (N*H*W, C)
#         query_norm_4 = F.normalize(data1_4_4_query, dim=1)
#         dot_norm_4 = F.normalize(self.dot_w_4, dim=1)
#         s = torch.mm(query_norm_4, dot_norm_4.transpose(0, 1))
#         dot_s_vec_4 = F.softmax(s, dim=1)
#         data_4_4_feature = torch.mm(dot_s_vec_4, self.dot_w_4)
#         data_4_4_feature = torch.reshape(data_4_4_feature, (-1, query_h, query_w, query_c)) # make (N, H, W, C)
#         data_4_4_feature = data_4_4_feature.permute(0, 3, 1, 2)
# #         print(data_4_4_feature.shape)
#         dataList.append(data_4_4_feature)
    
        return dataList


class outputStack(nn.Module):  # 输出层
    def __init__(self,input_channel):
        super(outputStack, self).__init__()
        self.BN = nn.BatchNorm2d(input_channel)
        self.relu = nn.Tanh()
        self.conv1 = SpectralNorm(nn.Conv2d(input_channel, 7, 1,1,0))


    def forward(self, x):

        x=self.BN(x)
        x=self.relu(x)
        x=self.conv1(x)
#         out=depth_to_space(x,2)

        return x


class outputStack_fuse(nn.Module):  # 输出层
    def __init__(self,input_channel):
        super(outputStack_fuse, self).__init__()
        self.BN = nn.BatchNorm2d(input_channel)
        self.relu = nn.Tanh()
        self.conv1 = SpectralNorm(nn.Conv2d(input_channel, 256, 1,1,0))


    def forward(self, x):

        x=self.BN(x)
        x=self.relu(x)
        x=self.conv1(x)
#         out=depth_to_space(x,2)

        return x


class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Softmax(),
        )

    def forward(self, inputs):
        b, c, _, _ = inputs.shape
        x = self.pool(inputs).view(b, c)
        x = self.net(x).view(b, c, 1, 1)
        x = inputs * x
        return x


class Stem_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=9, stride=stride, padding=4),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=9, padding=4),
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        # print("Stem_Block x", x.shape)
        # print("Stem_Block s", s.shape)
        # print("Stem_Block y", y.shape)
        return y


class ResNet_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            SpectralNorm(nn.Conv2d(in_c, out_c, kernel_size=9, padding=4, stride=stride)),
#             nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=9, padding=4)
        )

        self.c2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0)),
#             nn.BatchNorm2d(out_c),
        )

        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        x = self.c1(inputs)
        s = self.c2(inputs)
        y = self.attn(x + s)
        return y


class ASPP(nn.Module):
    def __init__(self, in_c, out_c, rate=[1, 6, 12, 18]):
        super().__init__()

        self.c1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[0], padding=rate[0])),
#             nn.BatchNorm2d(out_c)
        )

        self.c2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[1], padding=rate[1])),
#             nn.BatchNorm2d(out_c)
        )

        self.c3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[2], padding=rate[2])),
#             nn.BatchNorm2d(out_c)
        )

        self.c4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[3], padding=rate[3])),
#             nn.BatchNorm2d(out_c)
        )

        self.c5 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)


    def forward(self, inputs):
        x1 = self.c1(inputs)
        x2 = self.c2(inputs)
        x3 = self.c3(inputs)
        x4 = self.c4(inputs)
        x = x1 + x2 + x3 + x4
        y = self.c5(x)
        
        return y


class Attention_Block(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        out_c = in_c[1]

        self.g_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(),
            nn.Conv2d(in_c[0], out_c, kernel_size=9, padding=4),
#             nn.MaxPool2d((2, 2))
        )

        self.x_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(in_c[1], out_c, kernel_size=9, padding=4),
#             nn.MaxPool2d((2, 2))
        )

        self.gc_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=9, padding=4),
        )

    def forward(self, g, x):
        g_pool = self.g_conv(g)
        x_conv = self.x_conv(x)
        gc_sum = g_pool + x_conv
        gc_conv = self.gc_conv(gc_sum)
        y = gc_conv * x
        return y


class Decoder_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.a1 = Attention_Block(in_c)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.r1 = ResNet_Block(in_c[0]+in_c[1], out_c, stride=1)

    def forward(self, g, x):
        d = self.a1(g, x)
#         d = self.up(d)
        d = torch.cat([d, g], axis=1)
        d = self.r1(d)
        return d


class RobustMeanInterpolate(nn.Module):
    def __init__(self, channels, max_kernel_size=5):
        super(RobustMeanInterpolate, self).__init__()
        self.channels = channels
        self.max_kernel_size = max_kernel_size

    def forward(self, x):
        original_x = x.clone()  # 保存原始数据
        batch_size, channels, height, width = x.size()
#         padding = self.max_kernel_size // 2

        # 使用全局平均值填充
        global_means = torch.nanmean(x, dim=[2, 3], keepdim=True)

        for kernel_size in range(1, self.max_kernel_size + 1, 2):
#             print(kernel_size)
            padding = (kernel_size-1)//2
            kernel = torch.ones((self.channels, 1, kernel_size, kernel_size), device=x.device, dtype=x.dtype)
            kernel /= kernel_size ** 2

            x_padded = F.pad(original_x, (padding, padding, padding, padding), mode='replicate')
            is_not_nan = ~torch.isnan(x_padded).bool()  # 标记非NaN位置
            counts = F.conv2d(is_not_nan.float(), kernel, padding=0, groups=self.channels)  # 计算非NaN邻居数量
            sums = F.conv2d(torch.nan_to_num(x_padded), kernel, padding=0, groups=self.channels)
            means = sums / counts

            # 这里添加额外的填充以确保尺寸匹配
            means_padded = means
#             F.pad(means, (padding, padding, padding, padding), mode='constant', value=float('nan'))

            # 只在原始数据是NaN并且当前尺寸有非零count的位置更新
            valid_means = (counts > 0)
#             print(torch.isnan(x).shape)
#             print(valid_means.shape)
            x = torch.where(torch.isnan(x) & valid_means, means_padded, x)

        # 对于仍然是NaN的位置，使用全局平均值
        x = torch.where(torch.isnan(x), global_means.expand_as(x), x)
        return x
