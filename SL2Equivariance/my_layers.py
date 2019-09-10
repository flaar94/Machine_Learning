import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import product
import torchvision
import torchvision.transforms as transforms
from torch import Tensor
import numpy as np

from SL2_classes import Vertex, Graph, SL2Element, CayleyGraph
from time import time

INIT_ADJUST = 0.1


class LatticeLocalSL2(nn.Module):
    """
    Applies a locally-SL(2,Z)-equivariant convolution layer to a lattice
    """
    max_filter_dimensions: torch.tensor
    padding: torch.tensor
    filter_size: torch.tensor
    in_channels: int
    out_channels: int
    cayley_length: int
    indexing_dict: dict
    flat_indices: torch.tensor

    def __init__(self, in_channels: int, out_channels: int, filter_size, radius: int, len_fun: str = "len",
                 group: str = 'SL2', pad_type: str = "same"):
        """
        Note: All group related calculations are performed here to improve performance. The end result of this is the
        flat_indices attribute which tells the forward method where to put the weights to make the filter used for
        convolution

        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param filter_size: scalar length and width of the filter in the spacial dimensions
        :param pad_type: defines the type of padding. Allowable inputs are 'same', 'valid', and 'partial'. Partial pads
        the input as if there was no distortion, thus reducing the output by the same amount as a valid
        non-equivariant convolutional network
        :param radius: radius of the ball in the N-action where the action is defined
        """
        super().__init__()
        self.bias = nn.Parameter(torch.rand(out_channels, requires_grad=True))
        self.filter_size = self.bias.new_tensor([filter_size, filter_size], dtype=torch.long) if isinstance(
            filter_size, int) else self.weight.new_tensor(filter_size, dtype=torch.long)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, *self.filter_size,
                                               dtype=torch.float) * INIT_ADJUST)
        self.in_channels = in_channels
        self.out_channels = out_channels
        cayley = CayleyGraph(radius, len_fun=len_fun, filter_size=self.filter_size, group=group)
        self.cayley_length = len(cayley)
        self.max_filter_dimensions = self.weight.new_tensor(cayley.max_filter_index(), dtype=torch.long)
        if pad_type == "valid":
            self.padding = torch.tensor([0, 0], dtype=torch.long)
        elif pad_type == "same":
            self.padding = (self.max_filter_dimensions - 1) // 2
        elif pad_type == "partial":
            self.padding = (self.max_filter_dimensions - 1) // 2 - (self.filter_size - 1) // 2
        else:
            raise Exception(f"{pad_type} is an invalid pad_type")
        self.indexing_dict = {}
        for i, x in enumerate(cayley):
            self.indexing_dict[x] = i
        self.flat_indices = self.weight.new_empty(
            [self.out_channels, self.cayley_length, self.in_channels, *self.filter_size],
            dtype=torch.long, requires_grad=False)
        for i, g, j, k, ell in product(range(self.out_channels), cayley, range(self.in_channels),
                                       range(self.filter_size[0].item()), range(self.filter_size[1].item())):
            centered_ur_index = torch.tensor(
                [k - (self.filter_size[0].item() - 1) // 2, ell - (self.filter_size[0].item() - 1) // 2])
            distorted_index = g.inv_matrix() @ centered_ur_index
            distorted_index.to(dtype=torch.long)
            shifted_index = distorted_index + self.max_filter_dimensions // 2
            num = np.ravel_multi_index(
                (i, j, self.indexing_dict[g], shifted_index[0].item(), shifted_index[1].item()),
                (self.out_channels, self.in_channels, self.cayley_length,
                 *self.max_filter_dimensions))
            self.flat_indices[i, self.indexing_dict[g], j, k, ell] = int(num)
        self.flat_indices = self.flat_indices.view(-1)
        self.max_filter_dimensions = tuple(self.max_filter_dimensions)

    def forward(self, x):
        """
        Computes the SL(2,Z) locally equivariant convolution.

        :param x: Input vector
        :return indexing_dict, output_tensor: returns a dictionary connecting group elements to indices along with the
        pytorch tensor image of shape (batch_size, self.out_channels, size of local group,
                                           x-coordinate size, y-coordinate size)
        """
        temp_weight = self.weight.unsqueeze(1)
        temp_weight = temp_weight.repeat(1, self.cayley_length, 1, 1, 1)
        conv_filter = self.weight.new_zeros([self.out_channels, self.cayley_length, self.in_channels,
                                             *self.max_filter_dimensions], requires_grad=False)
        temp_weight = temp_weight.view(-1)
        conv_filter.put_(self.flat_indices, temp_weight)
        conv_filter = conv_filter.view(self.out_channels * self.cayley_length, self.in_channels,
                                       *self.max_filter_dimensions)
        output_tensor = F.conv2d(x, conv_filter, padding=tuple(self.padding))
        output_tensor = output_tensor.view(output_tensor.shape[0], self.out_channels, self.cayley_length,
                                           output_tensor.shape[2], output_tensor.shape[3])
        output_tensor += self.bias.view(1, -1, 1, 1, 1)
        return self.indexing_dict, output_tensor


class GroupLocalSL2(nn.Module):
    """
    Applies a locally-SL(2,Z)-equivariant convolution layer to functions on the (local) group
    """

    def __init__(self, in_channels: int, out_channels: int, filter_size,
                 input_radius: int, len_fun: str = "len", group_filter_radius: int = 1, group: str = 'SL2'):
        """

        :param in_channels:
        :param out_channels:
        :param filter_size:
        :param input_radius:
        :param len_fun:
        :param group_filter_radius: options are 'same', 'valid, or 'partial'
        :param group: options are 'SL2' or 'SL2pm'
        """
        super().__init__()
        self.filter_size = (filter_size, filter_size) if isinstance(filter_size, int) else tuple(filter_size)
        self.filter_cayley = CayleyGraph(group_filter_radius, len_fun=len_fun, group=group)
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, len(self.filter_cayley), *self.filter_size,
                                               dtype=torch.float) * INIT_ADJUST)
        self.bias = nn.Parameter(torch.rand(out_channels, requires_grad=True))
        output_radius = input_radius - group_filter_radius
        self.output_cayley = CayleyGraph(output_radius, len_fun=len_fun, group=group)
        self.out_index_dict = {}
        for i, x in enumerate(self.output_cayley):
            self.out_index_dict[x] = i

    def forward(self, x: torch.tensor, in_index_dict: dict):
        """

        :param x: pytorch tensor shape (batch_size, self.in_channels, size of input local group, x-coordinate size,
                                                                                                    y-coordinate size)
        :param in_index_dict: the indexing dictionary from the previous local G-CNN layer
        :return: (batch_size, self.out_channels, size of local group,
                                           x-coordinate size, y-coordinate size)
        """
        lst = []
        for i, g in enumerate(self.output_cayley):
            indices = [in_index_dict[g * h] for h in self.filter_cayley]
            temp_weight = self.weight.new_zeros(
                [self.weight.shape[0], self.weight.shape[1], x.shape[2], *self.filter_size])
            temp_weight[:, :, indices, :, :] = self.weight
            lst.append(temp_weight)
        stacked_weight = torch.cat(lst, 0)
        output_tensor = F.conv3d(x, stacked_weight)
        output_tensor = output_tensor.view(output_tensor.shape[0], self.weight.shape[0], len(self.output_cayley),
                                           output_tensor.shape[-2], output_tensor.shape[-1])
        output_tensor += self.bias.view(1, -1, 1, 1, 1)
        return self.out_index_dict, output_tensor


class GroupMaxPool(nn.MaxPool2d):
    def forward(self, x):
        channels = x.shape[1]
        cayley_size = x.shape[2]
        x = x.view(x.shape[0], channels * cayley_size, x.shape[-2], x.shape[-1])
        x = super().forward(x)
        x = x.view(x.shape[0], channels, cayley_size, x.shape[-2], x.shape[-1])
        return x


class GroupReLU(nn.ReLU):
    def forward(self, x):
        channels = x.shape[1]
        cayley_size = x.shape[2]
        x = x.view(x.shape[0], channels * cayley_size, x.shape[-2], x.shape[-1])
        x = super().forward(x)
        x = x.view(x.shape[0], channels, cayley_size, x.shape[-2], x.shape[-1])
        return x
