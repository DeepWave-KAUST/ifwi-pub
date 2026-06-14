"""ifwi_sweep.py -- minimal implicit FWI on the sweep wave solver.

Reproduction helpers for two implicit-FWI papers:

  * "Implicit full waveform inversion with energy-weighted gradient" -- a wavefield
    energy-weighted gradient (pseudo-Hessian) preconditioner, Overthrust example.
    DOI: 10.3997/2214-4609.202510069
  * "Multiresolution hash encoding for high resolution implicit full waveform
    inversion" -- a multiresolution hash encoding, Marmousi example.
    DOI: 10.3997/2214-4609.202510109

The ONLY external dependency is sweep's differentiable acoustic solver
(``sweep.equations.Acoustic`` + ``sweep.propagator.torch.PropTorch``), driven
with ``impl='c'`` (compiled CUDA kernels) and boundary-saving ("bs") mode.
The SIREN network, the Instant-NGP multiresolution hash encoding and the
pseudo-Hessian gradient preconditioning are all implemented here in plain
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

from sweep.equations import Acoustic
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
        return feat.permute(1, 0, 2).reshape(*shp, self.L * self.F)


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


def run_cached(path, fn, *, signature):
    """Cache an inversion result dict ('vp'/'loss'/'rel') to ``path``, keyed by
    ``signature`` (any config tuple). Reloads instantly when the cache exists and
    its signature matches; otherwise runs ``fn()`` and saves. This lets figure /
    layout tweaks re-run without recomputing the (slow) inversion -- delete the
    file or change any config value to force a fresh run.
    """
    import os
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        if str(d["signature"]) == str(signature):
            print(f"  [cache] loaded {os.path.basename(path)} (delete it to recompute)")
            return {"vp": d["vp"], "loss": d["loss"], "rel": d["rel"]}
    out = fn()
    np.savez(path, vp=out["vp"], loss=np.asarray(out["loss"]),
             rel=np.asarray(out["rel"]), signature=str(signature))
    return out
