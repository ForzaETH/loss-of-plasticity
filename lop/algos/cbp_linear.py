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
    """Continual Backpropagation (CBP) layer as described in
    "Continual Backpropagation: A Method for Plasticity and Stability in Artificial Neural Networks"
    (Dohare et al., 2024, https://arxiv.org/abs/2306.13812).
    This module wraps one or more nn.Linear layers (producers) and one or more nn.Linear layers (consumers),
    and tracks the utility of each output feature (neuron) based on a specified utility metric. Features with
    low utility are periodically reinitialized to promote plasticity.

    Args:
        in_layer (Union[nn.Linear, List[nn.Linear]]): Single or list of Linear layers producing features.
        out_layer (Union[nn.Linear, List[nn.Linear]]): Single or list of Linear layers consuming features.
        ln_layer (nn.LayerNorm, optional): Optional LayerNorm layer applied after in_layer(s).
        bn_layer (nn.BatchNorm1d, optional): Optional BatchNorm1d layer applied after in_layer(s).
        replacement_rate (float): Fraction of eligible features to reinitialize per forward pass (default: 1e-4).
        maturity_threshold (int): Minimum age (in forward passes) before a feature is eligible for reinit (default: 100).
        init (str): Initialization scheme for reinitialized features ('default', 'xavier', 'lecun'; default: 'kaiming').
        act_type (str): Activation function type following the in_layer(s) for gain calculation ('relu', 'tanh', etc.; default: 'relu').
        util_type (str): Utility metric to track ('contribution' supported; default: 'contribution').
        decay_rate (float): Decay rate for exponential moving averages in utility calculation (0=no decay, default: 0).
    """
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

        # Buffers for selection & ages
        self.register_buffer("util", torch.zeros(self.total_out_features))
        self.register_buffer("ages", torch.zeros(self.total_out_features))
        self.register_buffer("accumulated_num_features_to_replace", torch.zeros(1))

        # === EMA state per unit (paper Eqs. 3–8) ===
        self.register_buffer("f_ema", torch.zeros(self.total_out_features))
        self.register_buffer("z_ema", torch.zeros(self.total_out_features))
        self.register_buffer("u_ema", torch.zeros(self.total_out_features))

        # Optional: debiased readouts for inspection (Eq. 4, 8)
        self.register_buffer("f_hat", torch.zeros(self.total_out_features))
        self.register_buffer("z_hat", torch.zeros(self.total_out_features))
        self.register_buffer("u_hat", torch.zeros(self.total_out_features))

        if self.replacement_rate > 0:
            self.register_full_backward_hook(call_reinit)
            self.register_forward_hook(log_features)

        # Use first input layer for bound calculation
        self.bound = get_layer_bound(self.in_layer[0], init, nn.init.calculate_gain(nonlinearity=act_type))

        # === NEW: last reinit “event buffer” ===
        self.last_reinit_indices = None            # Tensor[Idx] in global (concatenated) feature space
        self.last_reinit_mask = None               # Bool mask over total_out_features
        self.last_reinit_payload = None            # Dict of slices written during reinit

    def reinit_cbp_linear(self):
        """Reset CBP state (ages, util, EMAs, features)."""
        dev = self.in_layer[0].weight.device
        self.util = torch.zeros(self.total_out_features, device=dev)
        self.ages = torch.zeros(self.total_out_features, device=dev)
        self.accumulated_num_features_to_replace = torch.zeros(1, device=dev)
        # reset EMA state as well
        self.f_ema.zero_()
        self.z_ema.zero_()
        self.u_ema.zero_()
        self.f_hat.zero_()
        self.z_hat.zero_()
        self.u_hat.zero_()
        # clear last-event buffer
        self.last_reinit_indices = None
        self.last_reinit_mask = None
        self.last_reinit_payload = None

    def forward(self, _input):
        return _input

    def _reduce_over_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Reduce a tensor over all but the last dimension (typically batch dim)."""
        if x.ndim == 1:
            return x
        reduce_dims = tuple(range(x.ndim - 1))
        return x.mean(dim=reduce_dims)

    def _incoming_avg_abs_per_unit_concat(self) -> torch.Tensor:
        """Average of absolute incoming weights per output unit, concatenated over all input layers."""
        parts = []
        for layer in self.in_layer:
            parts.append(layer.weight.detach().abs().mean(dim=1))
        return torch.cat(parts, dim=0)

    def _outgoing_abs_sum_per_input_concat(self) -> torch.Tensor:
        """Sum of absolute outgoing weights per input feature, concatenated over all input layers."""
        per_head = []
        for out in self.out_layer:
            per_head.append(out.weight.detach().abs().sum(dim=0))
        return torch.stack(per_head, dim=0).sum(dim=0)

    def get_features_to_reinit(self):
        """Determine which features to reinitialize based on their utility.

        Returns:
            Tensor: 1D tensor of feature indices in the global concatenated space to reinitialize
        """
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

        if self.features is None:
            return torch.empty(0, dtype=torch.long, device=self.util.device)

        dev = self.util.device
        eta = float(self.decay_rate)
        eps = 1e-8

        h_mean = self._reduce_over_batch(self.features.detach().to(dev))

        if eta == 0.0:
            self.f_ema.copy_(h_mean)
            self.f_hat.copy_(h_mean)
        else:
            self.f_ema.mul_(eta).add_(h_mean, alpha=(1 - eta))
            bias_corr = 1.0 - (eta ** self.ages)
            self.f_hat = torch.where(bias_corr > 0, self.f_ema / bias_corr, self.f_ema)

        if eta == 0.0:
            abs_mean_corr = h_mean.abs()
        else:
            abs_mean_corr = (h_mean - self.f_hat).abs()

        outgoing_sum = self._outgoing_abs_sum_per_input_concat().to(dev)
        z_inst = abs_mean_corr * outgoing_sum
        if eta == 0.0:
            self.z_ema.copy_(z_inst)
            self.z_hat.copy_(z_inst)
        else:
            self.z_ema.mul_(eta).add_(z_inst, alpha=(1 - eta))
            z_bias = 1.0 - (eta ** self.ages)
            self.z_hat = torch.where(z_bias > 0, self.z_ema / z_bias, self.z_ema)

        incoming_avg = self._incoming_avg_abs_per_unit_concat().to(dev)
        adapt_inst = 1.0 / (incoming_avg + eps)

        y_inst = z_inst * adapt_inst
        if eta == 0.0:
            self.u_ema.copy_(y_inst)
            self.u_hat.copy_(y_inst)
        else:
            self.u_ema.mul_(eta).add_(y_inst, alpha=(1 - eta))
            u_bias = 1.0 - (eta ** self.ages)
            self.u_hat = torch.where(u_bias > 0, self.u_ema / u_bias, self.u_ema)

        self.util.data = self.u_hat

        low_util_indices = torch.topk(-self.util[eligible_feature_indices], num_new_features_to_replace)[1]
        features_to_replace = eligible_feature_indices[low_util_indices]
        return features_to_replace

    @torch.no_grad()
    def reinit_features(self, features_to_replace: torch.Tensor):
        """Reinitialize the specified features in-place.

        Args:
            features_to_replace (Tensor): 1D tensor of feature indices in the global concatenated space.
        """
        # Record if nothing to do
        if features_to_replace.numel() == 0:
            self.last_reinit_indices = torch.empty(0, dtype=torch.long, device=self.util.device)
            self.last_reinit_mask = torch.zeros(self.total_out_features, dtype=torch.bool, device=self.util.device)
            self.last_reinit_payload = {"in": [], "out": [], "bn": None, "ln": None}
            return

        dev = self.in_layer[0].weight.device
        # Build global mask once
        mask = torch.zeros(self.total_out_features, dtype=torch.bool, device=dev)
        mask[features_to_replace] = True

        # Prepare payload dict that captures the VALUES written during reinit
        payload = {"in": [], "out": [], "bn": None, "ln": None}

        # Reinit producer rows and zero consumer columns
        for i, layer in enumerate(self.in_layer):
            start = self.in_layer_offsets[i]
            end = start + self.in_sizes[i]
            local_idx = features_to_replace[(features_to_replace >= start) & (features_to_replace < end)] - start
            if local_idx.numel() == 0:
                continue

            # Reinit rows (producer)
            layer.weight.data[local_idx, :] = torch.empty_like(layer.weight.data[local_idx, :]).uniform_(-self.bound, self.bound)
            layer.bias.data[local_idx] = 0.0

            # Zero outgoing columns (consumers)
            cols = start + local_idx
            for h, out in enumerate(self.out_layer):
                out.weight.data[:, cols] = 0.0
                # capture what we just wrote
                payload["out"].append({
                    "head": h,
                    "columns": cols.detach().clone(),
                    "weight_cols": out.weight.data[:, cols].detach().clone()
                })

            # capture producer rows we just wrote
            payload["in"].append({
                "in_block": i,
                "local_idx": local_idx.detach().clone(),
                "weight_rows": layer.weight.data[local_idx, :].detach().clone(),
                "bias_rows": layer.bias.data[local_idx].detach().clone()
            })

            # Reset ages/EMAs for those units
            self.ages[start + local_idx] = 0
            self.f_ema[start + local_idx] = 0.0
            self.z_ema[start + local_idx] = 0.0
            self.u_ema[start + local_idx] = 0.0
            self.f_hat[start + local_idx] = 0.0
            self.z_hat[start + local_idx] = 0.0
            self.u_hat[start + local_idx] = 0.0

            # BN/LN slices if present
            if self.bn_layer is not None:
                self.bn_layer.bias.data[start + local_idx] = 0.0
                self.bn_layer.weight.data[start + local_idx] = 1.0
                self.bn_layer.running_mean.data[start + local_idx] = 0.0
                self.bn_layer.running_var.data[start + local_idx] = 1.0

            if self.ln_layer is not None:
                self.ln_layer.bias.data[start + local_idx] = 0.0
                self.ln_layer.weight.data[start + local_idx] = 1.0

        # Store BN/LN slices after loop (optional, only if they exist & any idx present)
        if self.bn_layer is not None and features_to_replace.numel() > 0:
            payload["bn"] = {
                "indices": features_to_replace.detach().clone(),
                "weight": self.bn_layer.weight.data[mask].detach().clone(),
                "bias":   self.bn_layer.bias.data[mask].detach().clone(),
                "running_mean": self.bn_layer.running_mean.data[mask].detach().clone(),
                "running_var":  self.bn_layer.running_var.data[mask].detach().clone(),
            }
        if self.ln_layer is not None and features_to_replace.numel() > 0:
            payload["ln"] = {
                "indices": features_to_replace.detach().clone(),
                "weight": self.ln_layer.weight.data[mask].detach().clone(),
                "bias":   self.ln_layer.bias.data[mask].detach().clone(),
            }

        # === Save “last event” ===
        self.last_reinit_indices = features_to_replace.detach().clone()
        self.last_reinit_mask = mask.detach().clone()
        self.last_reinit_payload = payload

    def reinit(self):
        features_to_replace = self.get_features_to_reinit()
        self.reinit_features(features_to_replace)

    # === NEW: fetch latest reinit update ===
    @torch.no_grad()
    def get_last_reinit_update(self, clear: bool = True, to_cpu: bool = True):
        """
        Returns a dict with:
          - 'indices': Tensor[Idx] in global concatenated feature space
          - 'mask':    Bool tensor over total_out_features
          - 'payload': Dict with producer rows ('in'), consumer columns ('out'), and optional BN/LN slices

        Args:
            clear: if True, clear the internal buffer after fetching (default: True)
            to_cpu: if True, move all tensors to CPU (default: True)
        Returns:
            None if no reinit happened since last fetch.
        """
        if self.last_reinit_indices is None:
            return None

        def _maybe_cpu(t):
            return t.cpu() if (to_cpu and isinstance(t, torch.Tensor)) else (t.clone() if isinstance(t, torch.Tensor) else t)

        # Deep-copy payload tensors (and optionally move to CPU)
        payload = {"in": [], "out": [], "bn": None, "ln": None}
        for item in self.last_reinit_payload["in"]:
            payload["in"].append({
                "in_block": item["in_block"],
                "local_idx": _maybe_cpu(item["local_idx"]),
                "weight_rows": _maybe_cpu(item["weight_rows"]),
                "bias_rows": _maybe_cpu(item["bias_rows"]),
            })
        for item in self.last_reinit_payload["out"]:
            payload["out"].append({
                "head": item["head"],
                "columns": _maybe_cpu(item["columns"]),
                "weight_cols": _maybe_cpu(item["weight_cols"]),
            })
        for k in ("bn", "ln"):
            if self.last_reinit_payload[k] is not None:
                payload[k] = {
                    "indices": _maybe_cpu(self.last_reinit_payload[k]["indices"]),
                    "weight":  _maybe_cpu(self.last_reinit_payload[k]["weight"]),
                    "bias":    _maybe_cpu(self.last_reinit_payload[k]["bias"]),
                }
                if k == "bn":
                    payload[k]["running_mean"] = _maybe_cpu(self.last_reinit_payload[k]["running_mean"])
                    payload[k]["running_var"]  = _maybe_cpu(self.last_reinit_payload[k]["running_var"])

        out = {
            "indices": _maybe_cpu(self.last_reinit_indices),
            "mask": _maybe_cpu(self.last_reinit_mask),
            "payload": payload,
        }

        if clear:
            self.last_reinit_indices = None
            self.last_reinit_mask = None
            self.last_reinit_payload = None

        return out

    @torch.no_grad()
    def apply_reinit_update(
        self,
        update: dict,
        *,
        reset_internal_state: bool = True,
        strict: bool = True,
        to_device: torch.device = None,
    ) -> int:
        """
        Apply an external CBP reinit update (from source CBPLinear.get_last_reinit_update)
        to THIS CBPLinear (typically on the target critic).

        Args:
            update: dict with keys {"indices", "mask", "payload": {"in":[], "out":[], "bn":?, "ln":?}}
                    exactly as returned by get_last_reinit_update(clear=False/True).
            reset_internal_state: also reset ages and EMA buffers for affected units.
            strict: if True, assert shape compatibility; if False, best-effort apply.
            to_device: force tensors to this device (defaults to this module's device).

        Returns:
            int: number of features that were updated (len(indices)).
        """
        if update is None:
            return 0

        device = to_device if to_device is not None else self.in_layer[0].weight.device

        def td(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        # Pull indices/mask
        if "indices" not in update or update["indices"] is None:
            return 0
        indices = td(update["indices"]).long()
        if indices.numel() == 0:
            return 0

        mask = td(update.get("mask", None))
        payload = update.get("payload", {})
        in_items = payload.get("in", [])
        out_items = payload.get("out", [])
        bn_payload = payload.get("bn", None)
        ln_payload = payload.get("ln", None)

        # ===== Apply producer rows (in_layer[i]) =====
        for item in in_items:
            i = int(item["in_block"])
            local_idx = td(item["local_idx"]).long()
            w_rows = td(item["weight_rows"])
            b_rows = td(item["bias_rows"])

            tgt_in = self.in_layer[i]
            if strict:
                if w_rows.shape[1] != tgt_in.weight.shape[1]:
                    raise ValueError(
                        f"Shape mismatch on in_layer[{i}]: source rows have in_features={w_rows.shape[1]}, "
                        f"target has in_features={tgt_in.weight.shape[1]}"
                    )
                if local_idx.numel() != w_rows.shape[0] or (tgt_in.bias is not None and b_rows.shape[0] != local_idx.numel()):
                    raise ValueError("Row count mismatch for in_layer weights/bias.")

            # Write rows
            tgt_in.weight.data[local_idx, :] = w_rows
            if tgt_in.bias is not None:
                tgt_in.bias.data[local_idx] = b_rows

        # ===== Apply consumer columns (out_layer[h]) =====
        for item in out_items:
            h = int(item["head"])
            cols = td(item["columns"]).long()
            w_cols = td(item["weight_cols"])
            tgt_out = self.out_layer[h]

            if strict:
                if w_cols.shape[0] != tgt_out.weight.shape[0] or w_cols.shape[1] != cols.numel():
                    raise ValueError(
                        f"Shape mismatch on out_layer[{h}] columns: "
                        f"got {tuple(w_cols.shape)} for {cols.numel()} columns, "
                        f"target out_features={tgt_out.weight.shape[0]}"
                    )

            # Write columns (these are typically zeros from the source reinit)
            tgt_out.weight.data[:, cols] = w_cols

        # ===== Optional: BN/LN slices =====
        if bn_payload is not None and self.bn_layer is not None:
            bn_idx = td(bn_payload["indices"]).long()
            self.bn_layer.weight.data[bn_idx] = td(bn_payload["weight"])
            self.bn_layer.bias.data[bn_idx]   = td(bn_payload["bias"])
            if "running_mean" in bn_payload and "running_var" in bn_payload:
                self.bn_layer.running_mean.data[bn_idx] = td(bn_payload["running_mean"])
                self.bn_layer.running_var.data[bn_idx]  = td(bn_payload["running_var"])

        if ln_payload is not None and self.ln_layer is not None:
            ln_idx = td(ln_payload["indices"]).long()
            self.ln_layer.weight.data[ln_idx] = td(ln_payload["weight"])
            self.ln_layer.bias.data[ln_idx]   = td(ln_payload["bias"])

        # ===== Reset CBP internal state for those units (recommended) =====
        if reset_internal_state:
            # We need global (concatenated) indices -> already provided in `indices`
            self.ages[indices] = 0
            self.f_ema[indices] = 0.0
            self.z_ema[indices] = 0.0
            self.u_ema[indices] = 0.0
            self.f_hat[indices] = 0.0
            self.z_hat[indices] = 0.0
            self.u_hat[indices] = 0.0

        return int(indices.numel())
