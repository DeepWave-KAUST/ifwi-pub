# ifwi-pub

Reproducible synthetic examples for the *Geophysics* paper

> **Accelerating High Resolution Implicit Full Waveform Inversion**<br>
> Shaowen Wang and Tariq Alkhalifah

All forward/adjoint modeling runs on the
[**sweep**](https://github.com/DeepWave-KAUST/sweep) differentiable acoustic
solver. Each notebook reproduces one of the paper's two synthetic experiments —
three inversions on one model and the comparison figures:

| Notebook | Model | Experiments |
|---|---|---|
| [`notebooks/pseudo_hessian_overthrust.ipynb`](notebooks/pseudo_hessian_overthrust.ipynb) | Overthrust | conventional FWI · iFWI (SIREN) · iFWI **+ pseudo-Hessian** |
| [`notebooks/hash_encoding_marmousi.ipynb`](notebooks/hash_encoding_marmousi.ipynb) | Marmousi | conventional FWI · iFWI (SIREN) · iFWI **+ hash encoding** |

The two experiments were first presented as EAGE extended abstracts, whose methods
this repository reproduces:
* *Implicit full waveform inversion with energy-weighted gradient* — DOI [10.3997/2214-4609.202510069](https://doi.org/10.3997/2214-4609.202510069) (pseudo-Hessian / Overthrust)
* *Multiresolution hash encoding for high resolution implicit full waveform inversion* — DOI [10.3997/2214-4609.202510109](https://doi.org/10.3997/2214-4609.202510109) (hash encoding / Marmousi)

## Method

The velocity model is reparameterized by a coordinate network and inverted
through the wave equation:

```
vp(grid) = vp_init + std · net(coords) + mean
```

* **Conventional FWI** optimizes the grid velocity directly.
* **Implicit FWI (iFWI)** makes `net` a SIREN MLP (the paper's baseline).
* **Pseudo-Hessian** preconditions the velocity-grid gradient by the
  source/receiver illumination — `g ← g / sqrt(s·r)` — before back-propagating it
  into the network. `s` and `r` are read directly off the sweep solver
  (`solver.source_illumination` / `solver.receiver_illumination`), which the
  compiled backend fills during the backward pass. (Overthrust example.)
* **Hash encoding** prepends a native Instant-NGP multiresolution hash-grid
  encoder to the SIREN. (Marmousi example.)

Everything except the wave solver is implemented from scratch in
[`src/ifwi_sweep.py`](src/ifwi_sweep.py): the SIREN network, the multiresolution
hash encoding (pure PyTorch — **not** tinycudann) and the pseudo-Hessian
preconditioner. The notebooks depend only on `sweep` plus `torch` / `numpy` /
`matplotlib`.

## Solver settings

All forward/adjoint modeling uses the compiled CUDA backend with boundary
saving:

```python
solver = PropTorch(Acoustic(spatial_order=..., device="cuda"), shape, dh, dt,
                   abcn=..., source_type=["h1"], receiver_type=["h1"], impl="c")
syn = solver(wavelet, sources, receivers, models=[vp], use_boundary_saving=True)  # 'bs' mode
```

## Layout

```
src/ifwi_sweep.py     solver wiring + SIREN + hash encoding + pseudo-Hessian + inversion loops
src/models.py         Overthrust & Marmousi true/smooth vp, embedded (zlib + base85, no .npy files)
notebooks/            pseudo_hessian_overthrust + hash_encoding_marmousi
figures/              comparison figures written by the notebooks (and cached results, gitignored)
tools/encode_models.py  regenerates src/models.py from .npy (only needed to update the models)
```

## Installation

A CUDA GPU is required — this repo drives the solver with `impl='c'` (the compiled
CUDA path). Install the **sweep** solver ([`sweep-solver`](https://github.com/DeepWave-KAUST/sweep))
from source, building its CUDA extension:

```bash
git clone https://github.com/DeepWave-KAUST/sweep.git
cd sweep
SWEEP_BUILD_CUDA=1 pip install -v ".[cuda]" --no-build-isolation
```

If the build can't auto-detect your GPU, set `TORCH_CUDA_ARCH_LIST` before the
`pip` command (e.g. `"7.0"` for V100, `"8.0"` for A100, `"8.9"` for RTX 6000 Ada).
Full notes are in the [sweep docs](https://deepwave-kaust.github.io/sweep/getting-started/installation/).

The notebooks additionally need `torch numpy matplotlib jupyter` (all present in a
sweep environment; see [`requirements.txt`](requirements.txt)). No model data files
are needed — the velocity models are embedded in [`src/models.py`](src/models.py).

## Running

With `sweep` installed, run the notebooks:

```bash
cd notebooks
jupyter lab            # open a notebook and run all cells
```

The notebooks ship with `SMOKE = False` (the full paper run — Overthrust 500
iterations, Marmousi 200) and their results already baked in. Set `SMOKE = True`
in the config cell for a quick reduced-iteration check. Inversion results are
cached under `figures/cache_*.npz`, so re-running only re-plots (seconds) — delete
the cache, or change any config value, to recompute. Observed data is generated
on the fly by the same solver, so no pre-computed data is needed.
