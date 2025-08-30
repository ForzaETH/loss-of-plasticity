import torch
from torch import nn
from math import sqrt
from typing import List, Union


def call_reinit(m, i, o):
    m.reinit()


def log_features(m, i, o):
    with torch.no_grad():
        inputs = i if isinstance(i, (list, tuple)) else (i,)
        if m.decay_rate == 0:
            m.features = torch.cat(inputs, dim=-1)
        else:
            new_feat = torch.cat(inputs, dim=-1)
            if m.features is None:
                m.features = (1 - m.decay_rate) * new_feat
            else:
                m.features = m.features * m.decay_rate + (1 - m.decay_rate) * new_feat


def get_layer_bound(layer, init, gain):
    if isinstance(layer, nn.Conv1d):
        return sqrt(1 / (layer.in_channels * layer.kernel_size[0]))
    elif isinstance(layer, nn.Conv2d):
        return sqrt(1 / (layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]))
    elif isinstance(layer, nn.Linear):
        if init == 'default':
            bound = sqrt(1 / layer.in_features)
        elif init == 'xavier':
            bound = gain * sqrt(6 / (layer.in_features + layer.out_features))
        elif init == 'lecun':
            bound = sqrt(3 / layer.in_features)
        else:
            bound = gain * sqrt(3 / layer.in_features)
        return bound


class CBPLinear(nn.Module):
    def __init__(
            self,
            in_layer: Union[nn.Linear, List[nn.Linear]],
            out_layer: Union[nn.Linear, List[nn.Linear]],
            ln_layer: nn.LayerNorm = None,
            bn_layer: nn.BatchNorm1d = None,
            replacement_rate=1e-4,
            maturity_threshold=100,
            init='kaiming',
            act_type='relu',
            util_type='contribution',
            decay_rate=0,
    ):
        super().__init__()

        # Handle multiple input layers
        if isinstance(in_layer, nn.Linear):
            self.in_layer = [in_layer]
        elif isinstance(in_layer, (list, nn.ModuleList)) and all(isinstance(l, nn.Linear) for l in in_layer):
            self.in_layer = in_layer
        else:
            raise ValueError("in_layer must be a Linear layer or a list/ModuleList of Linear layers")

        # Handle multiple output layers
        if isinstance(out_layer, nn.Linear):
            self.out_layer = [out_layer]
        elif isinstance(out_layer, (list, nn.ModuleList)) and all(isinstance(l, nn.Linear) for l in out_layer):
            self.out_layer = out_layer
        else:
            raise ValueError("out_layer must be a Linear layer or a list/ModuleList of Linear layers")

        self.ln_layer = ln_layer
        self.bn_layer = bn_layer
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.util_type = util_type
        self.decay_rate = decay_rate
        self.features = None

        # Total number of output features
        self.in_sizes = [layer.out_features for layer in self.in_layer]
        self.total_out_features = sum(self.in_sizes)

        # Offset for each layer
        self.in_layer_offsets = [0] + list(torch.cumsum(torch.tensor(self.in_sizes), dim=0)[:-1])

        # self.util = nn.Parameter(torch.zeros(self.total_out_features), requires_grad=False)
        # self.ages = nn.Parameter(torch.zeros(self.total_out_features), requires_grad=False)
        # self.accumulated_num_features_to_replace = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.register_buffer("util", torch.zeros(self.total_out_features))
        self.register_buffer("ages", torch.zeros(self.total_out_features))
        self.register_buffer("accumulated_num_features_to_replace", torch.zeros(1))

        if self.replacement_rate > 0:
            self.register_full_backward_hook(call_reinit)
            self.register_forward_hook(log_features)

        # Use first input layer for bound calculation
        self.bound = get_layer_bound(self.in_layer[0], init, nn.init.calculate_gain(nonlinearity=act_type))

    def reinit_cbp_linear(self):
        self.util = torch.zeros(self.total_out_features).to(self.in_layer[0].weight.device)
        self.ages = torch.zeros(self.total_out_features).to(self.in_layer[0].weight.device)
        self.accumulated_num_features_to_replace = torch.zeros(1).to(self.in_layer[0].weight.device)

    def forward(self, _input):
        return _input

    def get_features_to_reinit(self):
        self.ages += 1
        eligible_feature_indices = torch.where(self.ages > self.maturity_threshold)[0]
        if eligible_feature_indices.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.util.device)

        num_new_features_to_replace = self.replacement_rate * eligible_feature_indices.numel()
        self.accumulated_num_features_to_replace += num_new_features_to_replace
        if self.accumulated_num_features_to_replace < 1:
            return torch.empty(0, dtype=torch.long, device=self.util.device)

        num_new_features_to_replace = int(self.accumulated_num_features_to_replace.item())
        self.accumulated_num_features_to_replace -= num_new_features_to_replace

        # Compute output weight magnitudes and feature utilities
        output_weight_mag = torch.stack([layer.weight.data.abs().mean(dim=0) for layer in self.out_layer]).mean(dim=0)
        feature_magnitudes = self.features.abs().mean(dim=tuple(range(self.features.ndim - 1)))
        self.util.data = output_weight_mag * feature_magnitudes

        # Get features with lowest utility
        low_util_indices = torch.topk(-self.util[eligible_feature_indices], num_new_features_to_replace)[1]
        features_to_replace = eligible_feature_indices[low_util_indices]
        return features_to_replace

    def reinit_features(self, features_to_replace):
        if features_to_replace.numel() == 0:
            return

        with torch.no_grad():
            for i, layer in enumerate(self.in_layer):
                start = self.in_layer_offsets[i]
                end = start + self.in_sizes[i]
                local_idx = features_to_replace[(features_to_replace >= start) & (features_to_replace < end)] - start

                if local_idx.numel() == 0:
                    continue

                layer.weight.data[local_idx, :] = torch.empty_like(layer.weight.data[local_idx, :]).uniform_(-self.bound, self.bound)
                layer.bias.data[local_idx] = 0.0

                for out in self.out_layer:
                    out.weight.data[:, start + local_idx] = 0.0

                self.ages[start + local_idx] = 0

                if self.bn_layer is not None:
                    self.bn_layer.bias.data[start + local_idx] = 0.0
                    self.bn_layer.weight.data[start + local_idx] = 1.0
                    self.bn_layer.running_mean.data[start + local_idx] = 0.0
                    self.bn_layer.running_var.data[start + local_idx] = 1.0
                if self.ln_layer is not None:
                    self.ln_layer.bias.data[start + local_idx] = 0.0
                    self.ln_layer.weight.data[start + local_idx] = 1.0




    def reinit(self):
        features_to_replace = self.get_features_to_reinit()
        self.reinit_features(features_to_replace)


# import torch
# from torch import nn
# from math import sqrt
# from typing import List, Union


# def call_reinit(m, i, o):
#     m.reinit()


# def log_features(m, i, o):
#     # with torch.no_grad():
#     #     if m.decay_rate == 0:
#     #         m.features = i[0]
#     #     else:
#     #         if m.features is None:
#     #             m.features = (1 - m.decay_rate) * i[0]
#     #         else:
#     #             m.features = m.features * m.decay_rate + (1 - m.decay_rate) * i[0]
#     with torch.no_grad():
#         if m.decay_rate == 0:
#             m.features = torch.cat(i, dim=-1)  # assuming i is a tuple of layer outputs
#         else:
#             new_feat = torch.cat(i, dim=-1)
#             if m.features is None:
#                 m.features = (1 - m.decay_rate) * new_feat
#             else:
#                 m.features = m.features * m.decay_rate + (1 - m.decay_rate) * new_feat



# def get_layer_bound(layer, init, gain):
#     if isinstance(layer, nn.Conv2d):
#         return sqrt(1 / (layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]))
#     elif isinstance(layer, nn.Linear):
#         if init == 'default':
#             bound = sqrt(1 / layer.in_features)
#         elif init == 'xavier':
#             bound = gain * sqrt(6 / (layer.in_features + layer.out_features))
#         elif init == 'lecun':
#             bound = sqrt(3 / layer.in_features)
#         else:
#             bound = gain * sqrt(3 / layer.in_features)
#         return bound


# class CBPLinear(nn.Module):
#     def __init__(
#             self,
#             in_layer: Union[nn.Linear, List[nn.Linear]],
#             out_layer: Union[nn.Linear, List[nn.Linear]],
#             ln_layer: nn.LayerNorm = None,
#             bn_layer: nn.BatchNorm1d = None,
#             replacement_rate=1e-4,
#             maturity_threshold=100,
#             init='kaiming',
#             act_type='relu',
#             util_type='contribution',
#             decay_rate=0,
#     ):
#         super().__init__()
#         # if type(in_layer) is not nn.Linear:
#         #     raise Warning("Make sure in_layer is a weight layer")
#         if not (isinstance(in_layer, nn.Linear) or
#                 (isinstance(in_layer, (list, nn.ModuleList)) and all(isinstance(layer, nn.Linear) for layer in in_layer))):
#             raise Warning("Make sure in_layer is a Linear layer or a list/ModuleList of Linear layers")
#         if not (isinstance(out_layer, nn.Linear) or
#                 (isinstance(out_layer, (list, nn.ModuleList)) and all(isinstance(layer, nn.Linear) for layer in out_layer))):
#             raise Warning("Make sure out_layer is a Linear layer or a list/ModuleList of Linear layers")
#         """
#         Define the hyper-parameters of the algorithm
#         """
#         self.replacement_rate = replacement_rate
#         self.maturity_threshold = maturity_threshold
#         self.util_type = util_type
#         self.decay_rate = decay_rate
#         self.features = None
#         """
#         Register hooks
#         """
#         if self.replacement_rate > 0:
#             self.register_full_backward_hook(call_reinit)
#             self.register_forward_hook(log_features)

#         # self.in_layer = in_layer
#         self.in_layer = in_layer if isinstance(in_layer, (list, nn.ModuleList)) else [in_layer]
#         self.out_layer = out_layer
#         self.ln_layer = ln_layer
#         self.bn_layer = bn_layer
#         """
#         Utility of all features/neurons
#         """
#         # self.util = nn.Parameter(torch.zeros(self.in_layer.out_features), requires_grad=False)
#         # self.ages = nn.Parameter(torch.zeros(self.in_layer.out_features), requires_grad=False)
#         self.util = nn.Parameter(torch.cat([torch.zeros(layer.out_features) for layer in self.in_layer]).to(torch.float), requires_grad=False)
#         self.ages = nn.Parameter(torch.cat([torch.zeros(layer.out_features) for layer in self.in_layer]).to(torch.float), requires_grad=False)
#         self.in_layer_offsets = [0] + list(torch.cumsum(torch.tensor([layer.out_features for layer in self.in_layer]), dim=0).tolist())
#         self.accumulated_num_features_to_replace = nn.Parameter(torch.zeros(1), requires_grad=False)
#         """
#         Calculate uniform distribution's bound for random feature initialization
#         """
#         self.bound = get_layer_bound(layer=self.in_layer, init=init, gain=nn.init.calculate_gain(nonlinearity=act_type))

#     def forward(self, _input):
#         return _input

#     def get_features_to_reinit(self):
#         """
#         Returns: Features to replace
#         """
#         features_to_replace = torch.empty(0, dtype=torch.long, device=self.util.device)
#         self.ages += 1
#         """
#         Calculate number of features to replace
#         """
#         eligible_feature_indices = torch.where(self.ages > self.maturity_threshold)[0]
#         if eligible_feature_indices.shape[0] == 0:  return features_to_replace

#         num_new_features_to_replace = self.replacement_rate*eligible_feature_indices.shape[0]
#         self.accumulated_num_features_to_replace += num_new_features_to_replace
#         if self.accumulated_num_features_to_replace < 1:    return features_to_replace

#         num_new_features_to_replace = int(self.accumulated_num_features_to_replace)
#         self.accumulated_num_features_to_replace -= num_new_features_to_replace
#         """
#         Calculate feature utility
#         """
#         if isinstance(self.out_layer, nn.Linear):
#             output_weight_mag = self.out_layer.weight.data.abs().mean(dim=0)
#         elif isinstance(self.out_layer, (list, nn.ModuleList)):
#             weight_mags = [layer.weight.data.abs().mean(dim=0) for layer in self.out_layer]
#             output_weight_mag = torch.stack(weight_mags).mean(dim=0)
#         self.util.data = output_weight_mag * self.features.abs().mean(dim=[i for i in range(self.features.ndim - 1)])
#         """
#         Find features with smallest utility
#         """
#         new_features_to_replace = torch.topk(-self.util[eligible_feature_indices], num_new_features_to_replace)[1]
#         new_features_to_replace = eligible_feature_indices[new_features_to_replace]
#         features_to_replace = new_features_to_replace
#         return features_to_replace

#     def reinit_features(self, features_to_replace):
#         """
#         Reset input and output weights for low utility features
#         """
#         # with torch.no_grad():
#         #     num_features_to_replace = features_to_replace.shape[0]
#         #     if num_features_to_replace == 0: return
#         #     self.in_layer.weight.data[features_to_replace, :] *= 0.0
#         #     self.in_layer.weight.data[features_to_replace, :] += \
#         #         torch.empty(num_features_to_replace, self.in_layer.in_features, device=self.util.device).uniform_(-self.bound, self.bound)
#         #     self.in_layer.bias.data[features_to_replace] *= 0

#         #     if isinstance(self.out_layer, nn.Linear):
#         #         self.out_layer.weight.data[:, features_to_replace] = 0
#         #     elif isinstance(self.out_layer, (list, nn.ModuleList)):
#         #         for layer in self.out_layer:
#         #             layer.weight.data[:, features_to_replace] = 0
#         #     self.ages[features_to_replace] = 0
#         with torch.no_grad():
#             if features_to_replace.shape[0] == 0:
#                 return
#             for i, layer in enumerate(self.in_layer):
#                 start = self.in_layer_offsets[i]
#                 end = self.in_layer_offsets[i + 1]
#                 local_indices = features_to_replace[(features_to_replace >= start) & (features_to_replace < end)] - start
#                 if local_indices.numel() == 0:
#                     continue

#                 layer.weight.data[local_indices, :] = torch.empty_like(layer.weight.data[local_indices, :]).uniform_(-self.bound, self.bound)
#                 layer.bias.data[local_indices] = 0

#                 if isinstance(self.out_layer, nn.Linear):
#                     self.out_layer.weight.data[:, start + local_indices] = 0
#                 elif isinstance(self.out_layer, (list, nn.ModuleList)):
#                     for out in self.out_layer:
#                         out.weight.data[:, start + local_indices] = 0

#                 self.ages[start + local_indices] = 0


#             """
#             Reset the corresponding batchnorm/layernorm layers
#             """
#             if self.bn_layer is not None:
#                 self.bn_layer.bias.data[features_to_replace] = 0.0
#                 self.bn_layer.weight.data[features_to_replace] = 1.0
#                 self.bn_layer.running_mean.data[features_to_replace] = 0.0
#                 self.bn_layer.running_var.data[features_to_replace] = 1.0
#             if self.ln_layer is not None:
#                 self.ln_layer.bias.data[features_to_replace] = 0.0
#                 self.ln_layer.weight.data[features_to_replace] = 1.0

#     def reinit(self):
#         """
#         Perform selective reinitialization
#         """
#         features_to_replace = self.get_features_to_reinit()
#         self.reinit_features(features_to_replace)
