from typing import Callable

from compressai.entropy_models import EntropyModel
import numpy as np
import torch


class WeightEntropyModule(EntropyModel):
    """entropy module for network parameters
    width * [- (self.n_bins // 2 - 1), ..., -1, 0, 1, 2, ..., self.n_bins //  2 - 1]
    e.g.) n_bins = 56, pmf_lengths = 55

    cdf: 1 / (1 + alpha) * slab + alpha / (1 + alpha) * spike
      spike: N (0, width / 6)
      slab: N(0, sigma)

    quantization interval: width
    """

    def __init__(
        self, cdf: Callable, width: float = 5e-3, data_type: str = "float32", **kwargs
    ):
        super().__init__(**kwargs)
        self.cdf = cdf
        self.width: float = width
        self._tail_mass = 1e-9
        # self.width = 5e-4 # my test
        self.width = 5e-5 # my test
        # used for compression
        self.data_type = data_type

        self.register_buffer("_n_bins", torch.IntTensor())
        self.update(force=True)

    def update(self, force: bool = False) -> bool:
        if self._n_bins.numel() > 0 and not force:
            return False

        delta = self.width / 2
        # accept self.width * 10000 * interval difference at maximum
        intervals: torch.Tensor = torch.arange(1, 10001)
        upper = self._likelihood_cumulative(
            intervals * self.width + delta, stop_gradient=True
        )
        lower = self._likelihood_cumulative(
            -intervals * self.width - delta, stop_gradient=True
        )
        # (upper - lower) - (1 - self._tail_mass)
        diff: torch.Tensor = self._tail_mass - lower - (1 - upper)
        if not (diff >= 0).any():
            self._n_bins = intervals[-1]
        else:
            n_bins = intervals[diff.argmax()]
            # even value
            # self._n_bins = ((n_bins - 1) // 2 + 1) * 2
            self._n_bins = (torch.div(n_bins - 1, 2, rounding_mode="trunc") + 1) * 2
        self._n_bins = self._n_bins.reshape((1,))

        # bound = (self._n_bins - 1) // 2
        bound = torch.div(self._n_bins - 1, 2, rounding_mode="trunc")
        bound = torch.clamp(bound.int(), min=0)

        self._offset = -bound

        pmf_start = -bound
        pmf_length = 2 * bound + 1

        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)

        samples = samples[None, :] + pmf_start[:, None, None]

        half = self.width / 2

        lower = self._likelihood_cumulative(
            samples * self.width - half, stop_gradient=True
        )
        upper = self._likelihood_cumulative(
            samples * self.width + half, stop_gradient=True
        )
        pmf = upper - lower

        pmf = pmf[:, 0, :]
        tail_mass = lower[:, 0, :1] + (1 - upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    def quantize(self, w: torch.Tensor, mode: str, means=None) -> torch.Tensor:
        if mode == "noise":
            assert self.training
            # add uniform noise: [-self.width / 2, self.width / 2]
            noise = (torch.rand_like(w) - 0.5) * self.width
            return w + noise

        symbols: torch.Tensor = torch.round(w / self.width)
        if mode == "symbols":
            # bound: torch.Tensor = (self._n_bins - 1) // 2
            bound: torch.Tensor = torch.div(self._n_bins - 1, 2, rounding_mode="trunc")
            symbols = torch.min(torch.max(symbols, -bound), bound)
            return symbols.int()
        elif mode == "dequantize":
            w_bound: torch.Tensor = (self._n_bins - 1) * self.width / 2
            # clamp with (-w_bound, w_bound)
            w_hat: torch.Tensor = torch.min(
                torch.max(symbols * self.width, -w_bound), w_bound
            )
            return (w_hat - w).detach() + w
        else:
            raise NotImplementedError

    def dequantize(
        self, inputs: torch.Tensor, means=None, dtype: torch.dtype = torch.float
    ) -> torch.Tensor:
        outputs = (inputs * self.width).type(dtype)
        return outputs

    # modified from _logits_cumulative
    def _likelihood_cumulative(
        self, inputs: torch.Tensor, stop_gradient: bool
    ) -> torch.Tensor:
        if stop_gradient:
            with torch.no_grad():
                return self.cdf(inputs)
        else:
            return self.cdf(inputs)

    def _likelihood(self, inputs: torch.Tensor) -> torch.Tensor:
        delta = self.width / 2
        v0 = inputs - delta
        v1 = inputs + delta
        lower = self._likelihood_cumulative(v0, stop_gradient=False)
        upper = self._likelihood_cumulative(v1, stop_gradient=False)
        likelihood = upper - lower
        return likelihood

    def forward(self, x: torch.Tensor, training=None) -> tuple:
        if self.width == 0:
            outputs = x
            likelihood = torch.ones_like(x) * (2 ** -32)
            return outputs, likelihood

        if training is None:
            training = self.training

        if not torch.jit.is_scripting():
            # x from B x C x ... to C x B x ...
            perm = np.arange(len(x.shape))
            perm[0], perm[1] = perm[1], perm[0]
            # Compute inverse permutation
            inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
        else:
            # TorchScript in 2D for static inference
            # Convert to (channels, ... , batch) format
            perm = (1, 2, 3, 0)
            inv_perm = (3, 0, 1, 2)

        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)

        # Add noise or quantize
        outputs = self.quantize(values, "dequantize")
        outputs_ent = self.quantize(values, "noise") if self.training else outputs

        likelihood = self._likelihood(outputs_ent)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        dims = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(N, 1, *size[2:])

    def compress(self, x):
        if self.width == 0:
            strings = list()
            for i in range(len(x)):
                string = encode_array(x[i].flatten().cpu().numpy(), self.data_type)
                strings.append(string)
            return strings

        indexes = self._build_indexes(x.size())
        return super().compress(x, indexes)

    def decompress(self, strings, size):
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        if self.width == 0:
            xs = list()
            for string in strings:
                x = decode_array(string, self.data_type)
                x = torch.from_numpy(x.copy()).to(self._quantized_cdf.device)
                xs.append(x)
            xs = torch.stack(xs).float().reshape(output_size)
            return xs

        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        return super().decompress(strings, indexes, torch.float32)


def encode_array(x: np.ndarray, data_type: str) -> bytes:
    if data_type == "float32":
        return x.astype(np.float32).tobytes()
    if data_type == "float16":
        return x.astype(np.float16).tobytes()
    # Zou+, ISM 21
    elif data_type == "uint8":
        bias = x.min()
        x_ = x - bias
        scale: float = (255 / x_.max()).astype(np.float32)
        arr_qua = np.round(x_ * scale).astype(np.uint8)
        return arr_qua.tobytes() + bias.tobytes() + scale.tobytes()
    else:
        raise NotImplementedError


def decode_array(string: bytes, data_type: str) -> np.ndarray:
    if data_type == "float32":
        return np.frombuffer(string, dtype=np.float32)
    if data_type == "float16":
        return np.frombuffer(string, dtype=np.float16).astype(np.float32)
    # Zou+, ISM 21
    elif data_type == "uint8":
        arr = np.frombuffer(string[:-8], dtype=np.uint8)
        bias = np.frombuffer(string[-8:-4], dtype=np.float32)
        scale = np.frombuffer(string[-4:], dtype=np.float32)
        return arr / scale + bias
    else:
        raise NotImplementedError




from torch import distributions as D


class SpikeAndSlabCDF:
    def __init__(
        self, width: float = 5e-3, sigma: float = 5e-2, alpha: float = 1000
    ) -> None:
        self.alpha = alpha

        mean = torch.tensor(0.0)
        self.slab = D.Normal(mean, torch.tensor(sigma))
        if width != 0:
            self.spike = D.Normal(mean, torch.tensor(width / 6))
        else:
            self.spike = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        cdf_slab = self.slab.cdf(x)
        if self.spike is None:
            return cdf_slab
        else:
            cdf_spike = self.spike.cdf(x)
            return (cdf_slab + self.alpha * cdf_spike) / (1 + self.alpha)