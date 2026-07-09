"""ifwi_sweep.py -- minimal implicit FWI on the sweep wave solver.

Reproduction helpers for two implicit-FWI papers:

  * "Implicit full waveform inversion with energy-weighted gradient" -- a wavefield
    energy-weighted gradient (pseudo-Hessian) preconditioner, Overthrust example.
    DOI: 10.3997/2214-4609.202510069
  * "Multiresolution hash encoding for high resolution implicit full waveform
    inversion" -- a multiresolution hash encoding, Marmousi example.
    DOI: 10.3997/2214-4609.202510109
  * A high-resolution, variable-density (acoustic-impedance) example: a joint
    (vp, z) inversion of the full Marmousi2 model that combines the hash encoding
    with multiscale frequency continuation, using sweep's two-parameter
    ``AcousticVRZ`` equation.

The ONLY external dependency is sweep's differentiable acoustic solver
(``sweep.equations.Acoustic`` / ``sweep.equations.AcousticVRZ`` +
``sweep.propagator.torch.PropTorch``), driven with ``impl='c'`` (compiled CUDA
kernels) and boundary-saving ("bs") mode. The SIREN network, the Instant-NGP
multiresolution hash encoding, the pseudo-Hessian gradient preconditioning and
the multiscale zero-phase low-pass filter are all implemented here in plain
PyTorch -- no sweep-nn, no sweep-opt, no tinycudann.

Implicit parameterization:
    vp(grid) = vp_init + std * net(coords) + mean
Conventional FWI optimizes the grid velocity directly.

Pseudo-Hessian: the velocity-grid gradient is divided by the
square-root of the source x receiver illumination (sqrt(sill * rill)) and the
preconditioned gradient is then back-propagated into the network parameters.
sill / rill are read straight off the sweep solver
(``solver.source_illumination`` / ``solver.receiver_illumination``), which the
compiled backend fills during the backward pass whenever the model requires a
gradient.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from sweep.equations import Acoustic, AcousticVRZ
from sweep.propagator.torch import PropTorch

SEED = 19491001


# --------------------------------------------------------------------------- #
# Physics: wavelet, acquisition geometry, solver, observed data
# --------------------------------------------------------------------------- #
def ricker(nt, dt, fm, delay):
    """Ricker wavelet, shape (nt,)."""
    t = np.arange(nt, dtype=np.float32) * dt - delay
    a = (np.pi * fm * t) ** 2
    return ((1.0 - 2.0 * a) * np.exp(-a)).astype(np.float32)


def survey(nx, src_interval, *, rec_interval=1, src_depth=0, rec_depth=0):
    """Surface acquisition in grid points.

    Returns sources (ns, 2) and receivers (ns, nr, 2), each coordinate [x, z].
    """
    sx = np.arange(0, nx, src_interval)
    sources = np.stack([sx, np.full_like(sx, src_depth)], 1).astype(np.int64)
    rx = np.arange(0, nx, rec_interval)
    rec = np.stack([rx, np.full_like(rx, rec_depth)], 1).astype(np.int64)
    receivers = np.repeat(rec[None], len(sources), 0)
    return sources, receivers


def make_solver(shape, dh, dt, *, spatial_order, abcn, device,
                pml_type="cpmlr", free_surface=False):
    """Acoustic PropTorch solver on the compiled CUDA backend (impl='c')."""
    eq = Acoustic(spatial_order=spatial_order, device=device)
    return PropTorch(eq, shape=tuple(shape), dh=dh, dt=dt, abcn=abcn,
                     free_surface=free_surface, pml_type=pml_type,
                     source_type=["h1"], receiver_type=["h1"], impl="c")


def generate_observed(solver, wavelet, sources, receivers, vp_true, *, batch=16):
    """Forward-model clean observed data with the true model (no gradient)."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(sources), batch):
            sl = slice(i, i + batch)
            outs.append(solver(wavelet, sources[sl], receivers[sl],
                               models=[vp_true], use_boundary_saving=True).detach())
    return torch.cat(outs, 0)


def make_vrz_solver(shape, dh, dt, *, spatial_order, abcn, device,
                    pml_type="cpmlr", free_surface=False, impl="c"):
    """Variable-density acoustic (``AcousticVRZ``) PropTorch solver, impl='c'.

    Same wiring as make_solver but with sweep's two-parameter velocity-impedance
    equation: with buoyancy ``b = vp/z`` and modulus ``kappa = z*vp`` it advances
    ``p_tt = kappa * (b*lap(p) + grad(b).grad(p))`` (so ``kappa*b = vp**2`` and the
    density enters only through ``grad(ln z)``). ``z`` is the acoustic impedance
    (rho*vp). The forward pass takes ``models=[vp, z]`` and back-propagates to both.
    """
    eq = AcousticVRZ(spatial_order=spatial_order, device=device)
    return PropTorch(eq, shape=tuple(shape), dh=dh, dt=dt, abcn=abcn,
                     free_surface=free_surface, pml_type=pml_type,
                     source_type=["h1"], receiver_type=["h1"], impl=impl)


def generate_observed_vrz(solver, wavelet, sources, receivers, vp_true, z_true, *, batch=16):
    """Forward-model clean observed data with the true (vp, z) model (no gradient)."""
    outs = []
    with torch.no_grad():
        for i in range(0, len(sources), batch):
            sl = slice(i, i + batch)
            outs.append(solver(wavelet, sources[sl], receivers[sl],
                               models=[vp_true, z_true], use_boundary_saving=True).detach())
    return torch.cat(outs, 0)


_FILT_CACHE = {}


def _butter_ba(dt, fc, order):
    """Digital scipy Butterworth ``(b, a)`` for a band -- ``(lo, hi)`` band-pass or scalar
    low-pass -- cached by ``(dt, fc, order)``."""
    key = (float(dt), tuple(fc) if isinstance(fc, (tuple, list)) else float(fc), int(order))
    if key not in _FILT_CACHE:
        from scipy import signal as _sps
        fs = 1.0 / dt
        if isinstance(fc, (tuple, list)):
            b, a = _sps.butter(order, Wn=[float(fc[0]), float(fc[1])], btype="bandpass", fs=fs)
        else:
            b, a = _sps.butter(order, Wn=float(fc), btype="lowpass", fs=fs)
        _FILT_CACHE[key] = (np.asarray(b, np.float64), np.asarray(a, np.float64))
    return _FILT_CACHE[key]


def lowpass_zerophase(x, dt, fc, order=3, time_axis=1):
    """Zero-phase Butterworth low-pass / band-pass via ``torchaudio`` forward-backward IIR
    (``torchaudio.functional.filtfilt``) -- differentiable, GPU-native, and matched to scipy
    ``sosfiltfilt`` / the reference ``filter_jax`` (``jax_filtfilt``) to ~1.8%.

    NOTE: a frequency-domain ``|H(f)|**2`` multiply (the earlier implementation, and
    sweep-preproc's ``bandpass_torch``) does NOT match ``filtfilt`` -- it is a *circular*
    convolution that folds the trace tail back to t=0 as a coherent artefact and is ~7% off
    for a narrow band even after zero-padding; that artefact injects a spurious near-source
    gradient that destabilises the joint (vp, z) inversion. The time-domain IIR here does
    not. ``fc``: ``(lo, hi)`` band-pass, scalar low-pass, or ``None`` (pass-through). Sweep
    receiver gathers are (nshots, nt, nrec, 1), hence the default ``time_axis=1``.
    """
    if fc is None:
        return x
    from torchaudio.functional import filtfilt as _filtfilt
    b, a = _butter_ba(dt, fc, order)
    aT = torch.as_tensor(a, device=x.device, dtype=torch.float64)
    bT = torch.as_tensor(b, device=x.device, dtype=torch.float64)
    xm = x.movedim(time_axis, -1).to(torch.float64)  # fp64: the narrow-band IIR recursion is
    y = _filtfilt(xm, aT, bT, clamp=False)            # unstable in fp32 (poles near |z|=1 -> NaN)
    return y.movedim(-1, time_axis).to(x.dtype)


def _fc_tag(fc):
    """Numeric tag for a multiscale cutoff -- the high-cut (Hz) for a band-pass
    pair, the cutoff itself for a scalar low-pass. Used to log the active band."""
    return float(fc[1]) if isinstance(fc, (tuple, list)) else float(fc)


def _fc_str(fc):
    """Human label for a multiscale cutoff: 'all' | '2-8' | '8'."""
    if fc is None:
        return "all"
    if isinstance(fc, (tuple, list)):
        return f"{fc[0]:g}-{fc[1]:g}"
    return f"{fc:g}"


# --------------------------------------------------------------------------- #
# Instant-NGP multiresolution hash encoding (native PyTorch)
# --------------------------------------------------------------------------- #
class MultiResHashGrid(nn.Module):
    """Multiresolution hash-grid encoding (Mueller et al. 2022, SIGGRAPH).

    Maps coords in ``[0, 1]^dim`` to features ``(..., n_levels * n_features)``.
    Levels whose dense grid fits the hash table use exact tiled indexing; finer
    levels fall back to spatial hashing with Knuth-style primes.
    DOI: https://doi.org/10.1145/3528223.3530127
    """

    def __init__(self, dim=2, *, n_levels=4, n_features_per_level=1,
                 log2_hashmap_size=18, base_resolution=64, finest_resolution=256,
                 res_eps=0.0):
        super().__init__()
        self.dim, self.L, self.F = dim, n_levels, n_features_per_level
        self.T = 2 ** log2_hashmap_size
        self.output_dim = self.L * self.F
        self.active_levels = n_levels   # progressive coarse->fine gate (see set_active_levels)
        growth = math.exp((math.log(finest_resolution) - math.log(base_resolution))
                          / max(1, self.L - 1))
        scales, offsets, strides, first_hash = [], [0], [], 0
        for lvl in range(self.L):
            s = base_resolution * (growth ** lvl) - 1.0
            res = int(math.ceil(s - res_eps) + 1)   # res_eps absorbs FP epsilon at integer boundaries
            scales.append(s)
            prod = res ** dim
            if prod <= self.T:
                n_entries, first_hash = ((prod + 7) // 8) * 8, first_hash + 1
            else:
                n_entries = self.T
            offsets.append(offsets[-1] + n_entries)
            strides.append([res ** d for d in range(dim)])
        self.first_hash = first_hash
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32))
        self.register_buffer("strides", torch.tensor(strides, dtype=torch.int64))
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.int64))
        self.register_buffer("primes", torch.tensor([1, 2654435761, 805459861],
                                                    dtype=torch.int64), persistent=False)
        corners = ([[0, 0], [0, 1], [1, 0], [1, 1]] if dim == 2 else
                   [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
        self.register_buffer("corners", torch.tensor(corners, dtype=torch.float32))
        self.latents = nn.Parameter(torch.empty(offsets[-1], self.F).uniform_(-1e-4, 1e-4))

    def _indices(self, vert):
        """vert: (L, N, 2**dim, dim) integer vertex coords -> (L, N, 2**dim) rows."""
        dense, parts = self.first_hash, []
        if dense > 0:
            parts.append((vert[:dense] * self.strides[:dense].view(-1, 1, 1, self.dim)).sum(-1))
        if dense < self.L:
            v, m = vert[dense:], 0xFFFFFFFF
            idx = (v[..., 0] & m) ^ ((v[..., 1] * self.primes[1]) & m)
            if self.dim == 3:
                idx = idx ^ ((v[..., 2] * self.primes[2]) & m)
            parts.append(idx & m)
        idx = torch.cat(parts, 0) if len(parts) > 1 else parts[0]
        idx = (idx % self.T) + self.offsets[:-1].view(self.L, 1, 1)
        return idx.clamp(0, self.latents.shape[0] - 1)

    def forward(self, pos):
        shp = pos.shape[:-1]
        pos = pos.reshape(-1, self.dim)
        ps = pos[None] * self.scales[:, None, None]              # (L, N, dim)
        floor = torch.floor(ps)
        vert = (floor[..., None, :] + self.corners.view(1, 1, -1, self.dim)).to(torch.int64)
        frac = (ps - floor)[..., None, :]
        w = ((1.0 - self.corners) + (2.0 * self.corners - 1.0) * frac).clamp(0, 1).prod(-1)
        feat = (self.latents[self._indices(vert)] * w[..., None]).sum(-2)   # (L, N, F)
        if self.active_levels < self.L:                                     # progressive coarse->fine
            gate = (torch.arange(self.L, device=feat.device) < self.active_levels).to(feat.dtype)
            feat = feat * gate[:, None, None]
        return feat.permute(1, 0, 2).reshape(*shp, self.L * self.F)

    def set_active_levels(self, n):
        """Enable only the coarsest ``n`` levels (progressive coarse-to-fine); finer
        levels are gated to zero until unlocked. Coupling this to the multiscale
        frequency stages caps spurious high-wavenumber content early -- the stabilizing
        effect of a coarse-to-fine mesh, without changing the solver grid."""
        self.active_levels = int(max(1, min(self.L, n)))


# --------------------------------------------------------------------------- #
# SIREN network + implicit velocity reparameterization
# --------------------------------------------------------------------------- #
class Siren(nn.Module):
    """Sine-activated MLP (Sitzmann et al. 2020), matching the paper's JAX
    reference ``ifwijax.networks.MLP``: a first layer plus ``layers`` hidden
    layers (all sin(omega . x)) and a linear output. The first layer always uses
    the SIREN "is_first" init U(-1/in, 1/in) -- including the hash-encoded case.
    DOI: 10.48550/arXiv.2006.09661"""

    def __init__(self, in_dim, hidden, layers, omega, *, out_dim=1, bias=False):
        super().__init__()
        self.omega = omega
        dims = [in_dim] + [hidden] * (layers + 1) + [out_dim]   # first + `layers` hidden + linear out
        self.lins = nn.ModuleList(nn.Linear(dims[i], dims[i + 1], bias=bias)
                                  for i in range(len(dims) - 1))
        with torch.no_grad():
            for i, lin in enumerate(self.lins):
                fin = lin.weight.shape[1]
                if i == 0:
                    lin.weight.uniform_(-1.0 / fin, 1.0 / fin)          # SIREN is_first init
                else:
                    bound = math.sqrt(6.0 / fin) / omega                # SIREN hidden/output init
                    lin.weight.uniform_(-bound, bound)

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = torch.sin(self.omega * x)
        return x


def _coords(nz, nx, device, lo, hi):
    z = torch.linspace(lo, hi, nz, device=device)
    x = torch.linspace(lo, hi, nx, device=device)
    zz, xx = torch.meshgrid(z, x, indexing="ij")
    return torch.stack([zz, xx], -1).reshape(-1, 2)


def water_mask(shape, n_water, device="cuda"):
    """Update mask (nz, 1) that freezes the top ``n_water`` rows -- 0 in the water
    column, 1 below. With an implicit model ``init + std * net * mask`` this keeps the
    known (flat) water layer fixed at its initial value, which removes a spurious
    degree of freedom at the seabed and stabilizes the joint (vp, z) inversion.
    Broadcasts over x."""
    m = torch.ones(shape[0], 1, device=device)
    m[:n_water] = 0.0
    return m


class VelocityNet(nn.Module):
    """Coordinate network: vp(grid) = vp_init + std * net(coords) + mean.

    Pure SIREN when ``hash_cfg is None``; SIREN with an Instant-NGP hash
    front-end otherwise. Coordinates are normalized to [0, 1] (as in the
    paper) in both cases.
    """

    def __init__(self, shape, *, std, mean=0.0, hidden=128, layers=6, omega=30.0,
                 hash_cfg=None, device="cuda"):
        super().__init__()
        nz, nx = shape
        self.shape, self.std, self.mean = (nz, nx), float(std), float(mean)
        if hash_cfg is not None:
            self.encoder = MultiResHashGrid(2, **hash_cfg)
            in_dim = self.encoder.output_dim
        else:
            self.encoder, in_dim = None, 2
        # reference grid is [0, 1] for both pure-SIREN and hash inputs
        self.register_buffer("coords", _coords(nz, nx, device, 0.0, 1.0))
        self.siren = Siren(in_dim, hidden, layers, omega)
        self.to(device)

    def forward(self, vp_init):
        x = self.encoder(self.coords) if self.encoder is not None else self.coords
        out = self.siren(x).reshape(*self.shape)
        return vp_init + self.std * out + self.mean


class MultiParamNet(nn.Module):
    """Joint multiparameter coordinate network for variable-density (VRZ) iFWI.

    A single SIREN with one output channel per inverted parameter; channel ``i``
    maps to a grid model
        model_i(grid) = init_i + std_i * net(coords)[..., i] + mean_i
    matching the paper's JAX reference (``out = net; models = out*std + mean + init``).
    With ``hash_cfg`` an Instant-NGP hash front-end is prepended (bias-free SIREN, as
    in the reference). Coordinates are normalized to [0, 1]. Used for the joint
    (vp, z) high-resolution Marmousi2 example; ``std``/``mean`` are per-parameter
    lists, e.g. ``std=[4000, 10000], mean=[0, 0]``.
    """

    def __init__(self, shape, *, std, mean=None, hidden=128, layers=2, omega=10.0,
                 hash_cfg=None, mask=None, device="cuda"):
        super().__init__()
        nz, nx = shape
        self.shape = (nz, nx)
        self.std = [float(s) for s in std]
        self.mean = [float(m) for m in (mean if mean is not None else [0.0] * len(self.std))]
        self.k = len(self.std)
        if hash_cfg is not None:
            self.encoder = MultiResHashGrid(2, **hash_cfg)
            in_dim = self.encoder.output_dim
        else:
            self.encoder, in_dim = None, 2
        self.register_buffer("coords", _coords(nz, nx, device, 0.0, 1.0))
        # optional update mask (e.g. water_mask): net contribution is 0 where mask==0
        self.register_buffer("mask", None if mask is None else
                             torch.as_tensor(mask, dtype=torch.float32, device=device))
        self.siren = Siren(in_dim, hidden, layers, omega, out_dim=self.k)
        self.to(device)

    def forward(self, inits):
        """inits: list of ``k`` initial grids (tensors, each == self.shape).

        Returns a list of ``k`` reparameterized model grids in the same order. When a
        ``mask`` was given the network perturbation is gated by it, so masked-out cells
        (e.g. the water layer) stay pinned to their initial value.
        """
        x = self.encoder(self.coords) if self.encoder is not None else self.coords
        out = self.siren(x).reshape(*self.shape, self.k)
        if self.mask is not None:
            out = out * self.mask.unsqueeze(-1)      # (nz, 1) -> (nz, 1, 1) broadcasts over x, k
        return [inits[i] + self.std[i] * out[..., i] + self.mean[i] for i in range(self.k)]


# --------------------------------------------------------------------------- #
# Inversion driver
# --------------------------------------------------------------------------- #
def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _precondition(grad, scale, eps_ill, ill_water, grad_clip_q=0.0):
    """Energy-weight a velocity gradient by the illumination ``scale``.

    ill_water == 0 reproduces the reference exactly: ``grad / (scale + eps_ill)``.
    ill_water > 0 normalizes the illumination to its max and applies a relative
    'water level' floor -- ``grad / (scale/scale.max() + ill_water)`` -- which
    caps how much poorly-illuminated (dark) regions are amplified.
    grad_clip_q > 0 additionally clamps the preconditioned gradient to its
    [q, 1-q] quantiles, removing the rare outlier steps (e.g. a bad mini-batch
    hitting a dark region) without globally damping the preconditioner.
    """
    if ill_water > 0.0:
        scale = scale / (scale.max() + 1e-30)
        g = grad / (scale + ill_water)
    else:
        g = grad / (scale + eps_ill)
    if grad_clip_q > 0.0:
        lo = torch.quantile(g, grad_clip_q)
        hi = torch.quantile(g, 1.0 - grad_clip_q)
        g = g.clamp(lo, hi)
    return g


def run_inversion(solver, wavelet, sources, receivers, obs, model_fn, params, *,
                  epochs, lr, batch=8, eps=1e-22, use_ph=False, eps_ill=1e-11,
                  ill_water=0.0, grad_clip_q=0.0, per_shot_ill=True, true_vp=None,
                  seed=SEED, log_every=50, lr_decay=1.0):
    """Run one inversion. ``model_fn()`` returns the current vp grid as a
    function of ``params``; an Adam step is taken on ``params`` each epoch.

    use_ph=True applies the paper's pseudo-Hessian: the velocity-grid gradient is
    weighted by the source*receiver illumination before being back-propagated
    into ``params``. per_shot_ill=True (the paper setting) accumulates the scale
    per shot -- ``sum_shot sqrt(sill*rill)`` -- matching the reference exactly
    (one batched=1 solve per shot); per_shot_ill=False uses the cheaper batch
    approximation ``sqrt(sum sill * sum rill)`` from a single batched solve.
    """
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(params, lr=lr, eps=eps)
    sched = (torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)
             if lr_decay < 1.0 else None)
    ns = len(sources)
    hist = {"loss": [], "rel": []}
    for ep in range(epochs):
        idx = rng.choice(ns, size=min(batch, ns), replace=False)
        opt.zero_grad(set_to_none=True)
        vp = model_fn()
        if use_ph and per_shot_ill:
            # exact paper setting: (sum_shot grad) / (sum_shot sqrt(sill*rill)).
            # One batched=1 solve per shot gives that shot's own illumination.
            grad_acc, scale_acc, loss_value = torch.zeros_like(vp), torch.zeros_like(vp), 0.0
            for s in idx:
                syn = solver(wavelet, sources[s:s + 1], receivers[s:s + 1],
                             models=[vp], use_boundary_saving=True)
                loss_s = (syn - obs[s:s + 1]).pow(2).mean()
                (g_s,) = torch.autograd.grad(loss_s, vp, retain_graph=True)
                grad_acc = grad_acc + g_s
                scale_acc = scale_acc + torch.sqrt(
                    solver.source_illumination * solver.receiver_illumination)
                loss_value += float(loss_s.detach())
            loss_value /= len(idx)
            g_vp = _precondition(grad_acc, scale_acc, eps_ill, ill_water, grad_clip_q)
            for p, gp in zip(params, torch.autograd.grad(vp, params, grad_outputs=g_vp)):
                p.grad = gp
        else:
            syn = solver(wavelet, sources[idx], receivers[idx], models=[vp],
                         use_boundary_saving=True)
            loss = (syn - obs[idx]).pow(2).mean()
            loss_value = float(loss.detach())
            if use_ph:
                # batch approximation: (sum grad) / sqrt(sum sill * sum rill)
                (g_vp,) = torch.autograd.grad(loss, vp, retain_graph=True)
                scale = torch.sqrt(solver.source_illumination * solver.receiver_illumination)
                g_vp = _precondition(g_vp, scale, eps_ill, ill_water, grad_clip_q)
                for p, gp in zip(params, torch.autograd.grad(vp, params, grad_outputs=g_vp)):
                    p.grad = gp
            else:
                loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        hist["loss"].append(loss_value)
        if true_vp is not None:
            with torch.no_grad():
                hist["rel"].append(rel_l2(model_fn().detach().cpu().numpy(), true_vp))
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            msg = f"  ep {ep:4d}  loss {hist['loss'][-1]:.4e}"
            if true_vp is not None:
                msg += f"  model-relL2 {hist['rel'][-1]:.4f}"
            print(msg, flush=True)
    with torch.no_grad():
        hist["vp"] = model_fn().detach().cpu().numpy()
    return hist


def conventional_fwi(solver, wavelet, sources, receivers, obs, vp_init_t, **kw):
    """Experiment (1): optimize the grid velocity directly."""
    vp = vp_init_t.clone().requires_grad_(True)
    return run_inversion(solver, wavelet, sources, receivers, obs,
                         model_fn=lambda: vp, params=[vp], **kw)


def implicit_fwi(solver, wavelet, sources, receivers, obs, vp_init_t, vnet, *,
                 use_ph=False, **kw):
    """Experiments (2)/(3): implicit FWI; optional pseudo-Hessian (use_ph)."""
    return run_inversion(solver, wavelet, sources, receivers, obs,
                         model_fn=lambda: vnet(vp_init_t),
                         params=list(vnet.parameters()), use_ph=use_ph, **kw)


def _waveform_misfit(syn, obs, kind="l2", time_axis=1, eps=1e-8):
    """Data misfit: summed-L2 (``'l2sum'``), mean-L2 (``'l2'``), or trace-normalized
    cosine-similarity (``'cs'``).

    'l2sum' -- ``sum((syn - obs)**2)`` over every sample -- exactly matches the GJI
    FWIM reference (``jnp.sum((syn-obs)**2)``). The absolute scale matters: with
    Adam's ``eps=1e-22`` the mean-L2 gradient can be small enough that the second
    moment underflows in fp32 and the ``+eps`` denominator blows the step up, so the
    faithful reproduction uses 'l2sum'. 'cs' -- ``mean(1 - cos(syn, obs))`` along
    ``time_axis`` -- is amplitude-independent (compares waveform shape, not size).
    """
    syn = syn.reshape(obs.shape)
    if kind == "cs":
        num = (syn * obs).sum(dim=time_axis)
        den = syn.norm(dim=time_axis).clamp_min(eps) * obs.norm(dim=time_axis).clamp_min(eps)
        return (1.0 - num / den).mean()
    if kind == "l2sum":
        return (syn - obs).pow(2).sum()
    return (syn - obs).pow(2).mean()


def run_inversion_multiscale(solver, wavelet, sources, receivers, obs, net, inits, *,
                             epochs, lr, batch, freqs, dt, misfit="l2", hash_levels=None,
                             filt_order=3, time_axis=1, eps=1e-22, lr_decay=0.9995,
                             true_models=None, param_names=None, seed=SEED, log_every=50):
    """Joint multiparameter implicit FWI with multiscale frequency continuation.

    ``net(inits)`` returns a list of ``k`` model grids (e.g. ``[vp, z]``). Each
    epoch samples ``batch`` shots, forward-models them, low-pass filters BOTH the
    synthetic and the observed gathers at the current stage cutoff, and takes an
    Adam step (``eps``; per-step exponential ``lr_decay``) on the L2 waveform
    misfit. ``freqs`` is the list of stage cutoffs in Hz with ``None`` = unfiltered
    ('all'); the epochs are split evenly across the stages (``epochs // len(freqs)``),
    reproducing the reference multiscale schedule. ``misfit`` picks the L2 or
    trace-normalized cosine-similarity ('cs') misfit. ``hash_levels`` (a list aligned
    with ``freqs``) progressively unlocks hash-encoding levels stage by stage
    (coarse-to-fine), capping high-wavenumber content early to stabilize the
    high-resolution inversion. When ``true_models`` is given the
    per-parameter relative-L2 model error is tracked. Returns a dict with ``loss``,
    ``freq``, one array per parameter (named by ``param_names`` or ``m0``/``m1``/...)
    and ``rel_<name>`` error curves.
    """
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=eps)
    sched = (torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)
             if lr_decay < 1.0 else None)
    ns, k = len(sources), len(inits)
    names = list(param_names) if param_names else [f"m{i}" for i in range(k)]
    per = max(1, epochs // len(freqs))
    enc = getattr(net, "encoder", None)   # hash encoder, for progressive coarse->fine
    hist = {"loss": [], "freq": []}
    rels = {nm: [] for nm in names}
    snaps = {nm: [] for nm in names}; snap_freq = []   # per-stage-end model snapshots
    for ep in range(epochs):
        stage = min(ep // per, len(freqs) - 1)
        fc = freqs[stage]
        if hash_levels is not None and enc is not None:
            enc.set_active_levels(hash_levels[stage])
        idx = rng.choice(ns, size=min(batch, ns), replace=False)
        opt.zero_grad(set_to_none=True)
        models = net(inits)
        syn = solver(wavelet, sources[idx], receivers[idx], models=models,
                     use_boundary_saving=True)
        syn_f = lowpass_zerophase(syn, dt, fc, filt_order, time_axis)
        obs_f = lowpass_zerophase(obs[idx], dt, fc, filt_order, time_axis)
        loss = _waveform_misfit(syn_f, obs_f, misfit, time_axis)
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        hist["loss"].append(float(loss.detach()))
        hist["freq"].append(-1.0 if fc is None else float(fc))
        if true_models is not None:
            with torch.no_grad():
                for i, nm in enumerate(names):
                    rels[nm].append(rel_l2(models[i].detach().cpu().numpy(), true_models[i]))
        if ep + 1 == (stage + 1) * per or ep == epochs - 1:   # end of a frequency stage -> snapshot
            with torch.no_grad():
                sm = net(inits)
            for i, nm in enumerate(names):
                snaps[nm].append(sm[i].detach().cpu().numpy())
            snap_freq.append(-1.0 if fc is None else float(fc))
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            msg = f"  ep {ep:4d}  fc {('all' if fc is None else fc):>4}  loss {hist['loss'][-1]:.4e}"
            if true_models is not None:
                msg += "  relL2 " + " ".join(f"{nm}={rels[nm][-1]:.4f}" for nm in names)
            print(msg, flush=True)
    with torch.no_grad():
        final = net(inits)
    for i, nm in enumerate(names):
        hist[nm] = final[i].detach().cpu().numpy()
        hist[f"rel_{nm}"] = rels[nm]
        if snaps[nm]:
            hist[f"snap_{nm}"] = np.stack(snaps[nm])   # (n_stages, nz, nx)
    hist["snap_freq"] = snap_freq
    return hist


def run_ifwim_restart(solver, wavelet, sources, receivers, obs, net_factory, inits, *,
                      epochs_per_scale, lr, batch, freqs, dt, misfit="l2", filt_order=3,
                      time_axis=1, eps=1e-22, lr_decay=0.999, true_models=None,
                      param_names=None, seed=SEED, log_every=25, clamp=None,
                      checkpoint_path=None):
    """Scale-by-scale implicit FWI imaging with optimizer/network **restart** at each
    frequency scale -- the GJI FWIM paper scheme.

    For each cutoff in ``freqs`` a FRESH network + Adam are built from
    ``net_factory()`` (weights re-randomized, lr reset), the current model is
    reparameterized as ``net(cur_init)`` and trained for ``epochs_per_scale`` epochs at
    that scale's low-pass; the scale's inverted models then become ``cur_init`` for the
    next scale. This warm restart stops high-wavenumber artefacts from accumulating
    across scales, which is what stabilizes the joint (v, Z) inversion. ``net_factory``
    is a zero-arg callable returning a fresh ``MultiParamNet``; ``inits`` are the initial
    grids (e.g. the smooth [vp, z]).
    """
    rng = np.random.default_rng(seed)
    ns, k = len(sources), len(inits)
    names = list(param_names) if param_names else [f"m{i}" for i in range(k)]
    def _clamp(models):                                 # keep each model in its physical box
        if clamp is None:
            return models
        return [m if clamp[i] is None else m.clamp(clamp[i][0], clamp[i][1])
                for i, m in enumerate(models)]
    cur = [i.detach().clone() for i in inits]
    hist = {"loss": [], "freq": []}
    rels = {nm: [] for nm in names}
    snaps = {nm: [] for nm in names}                    # per-scale (end-of-band) model snapshots
    snap_freq = []
    for scale, fc in enumerate(freqs):
        net = net_factory()                             # fresh weights + fresh optimizer per scale
        opt = torch.optim.Adam(net.parameters(), lr=lr, eps=eps)
        sched = (torch.optim.lr_scheduler.ExponentialLR(opt, lr_decay)
                 if lr_decay < 1.0 else None)
        for ep in range(epochs_per_scale):
            idx = rng.choice(ns, size=min(batch, ns), replace=False)
            opt.zero_grad(set_to_none=True)
            models = _clamp(net(cur))
            syn = solver(wavelet, sources[idx], receivers[idx], models=models,
                         use_boundary_saving=True)
            loss = _waveform_misfit(
                lowpass_zerophase(syn, dt, fc, filt_order, time_axis),
                lowpass_zerophase(obs[idx], dt, fc, filt_order, time_axis), misfit, time_axis)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            hist["loss"].append(float(loss.detach()))
            hist["freq"].append(-1.0 if fc is None else _fc_tag(fc))
            if true_models is not None:
                with torch.no_grad():
                    for i, nm in enumerate(names):
                        rels[nm].append(rel_l2(models[i].detach().cpu().numpy(), true_models[i]))
            if log_every and (ep % log_every == 0 or ep == epochs_per_scale - 1):
                msg = (f"  scale {scale} fc {_fc_str(fc):>7} "
                       f"ep {ep:3d}  loss {hist['loss'][-1]:.3e}")
                if true_models is not None:
                    msg += "  relL2 " + " ".join(f"{nm}={rels[nm][-1]:.4f}" for nm in names)
                print(msg, flush=True)
        with torch.no_grad():                           # carry (bake) this scale's result into the next
            cur = [m.detach().clone() for m in _clamp(net(cur))]
            for i, nm in enumerate(names):              # snapshot the end-of-band model
                snaps[nm].append(cur[i].cpu().numpy())
            snap_freq.append(-1.0 if fc is None else _fc_tag(fc))
        if checkpoint_path is not None:                 # persist EACH band as it finishes (survives walltime kill)
            ckpt = {"snap_freq": np.asarray(snap_freq), "bands_done": scale + 1,
                    "loss": np.asarray(hist["loss"]), "freq": np.asarray(hist["freq"])}
            for i, nm in enumerate(names):
                ckpt[nm] = cur[i].detach().cpu().numpy()        # latest full model
                ckpt[f"snap_{nm}"] = np.stack(snaps[nm])        # (bands_done, nz, nx)
                if true_models is not None:
                    ckpt[f"rel_{nm}"] = np.asarray(rels[nm])
            np.savez(checkpoint_path, **ckpt)
            print(f"  [checkpoint] saved {scale + 1}/{len(freqs)} bands -> {checkpoint_path}", flush=True)
    for i, nm in enumerate(names):
        hist[nm] = cur[i].detach().cpu().numpy()
        hist[f"rel_{nm}"] = rels[nm]
        hist[f"snap_{nm}"] = np.stack(snaps[nm])
    hist["snap_freq"] = snap_freq
    return hist


def run_cached(path, fn, *, signature):
    """Cache an inversion result dict to ``path``, keyed by ``signature`` (any config
    tuple). Reloads instantly when the cache exists and its signature matches;
    otherwise runs ``fn()`` and saves every array/list it returns. This lets figure /
    layout tweaks re-run without recomputing the (slow) inversion -- delete the file
    or change any config value to force a fresh run. Works for both the single-model
    runs ('vp'/'loss'/'rel') and the multiparameter multiscale run
    ('vp'/'z'/'loss'/'rel_vp'/'rel_z'/...).
    """
    import os
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        if str(d["signature"]) == str(signature):
            print(f"  [cache] loaded {os.path.basename(path)} (delete it to recompute)")
            return {k: d[k] for k in d.files if k != "signature"}
    out = fn()
    np.savez(path, signature=str(signature),
             **{k: np.asarray(v) for k, v in out.items()})
    return out
