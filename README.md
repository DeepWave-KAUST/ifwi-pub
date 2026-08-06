# ifwi-pub

Reproducible synthetic examples for two companion papers by **Shaowen Wang and
Tariq Alkhalifah**:

> **Accelerating High Resolution Implicit Full Waveform Inversion** (*Geophysics*)<br>
> — the single-parameter velocity examples (Overthrust, Marmousi)
>
> **Implicit Full Waveform Inversion Imaging** (*GJI*, 2026) — DOI [10.1093/gji/ggag277](https://doi.org/10.1093/gji/ggag277)<br>
> — the joint velocity–impedance imaging (IFWIM) example (Marmousi2)

All forward/adjoint modeling runs on the
[**sweep**](https://github.com/DeepWave-KAUST/sweep) differentiable acoustic
solver. The two single-parameter notebooks each reproduce one *Geophysics* experiment —
three inversions on one model and the comparison figures — and the multiparameter notebook
reproduces the joint $(v_p, Z)$ imaging example of the *GJI* paper:

| Notebook | Model | Experiments |
|---|---|---|
| [`notebooks/pseudo_hessian_overthrust.ipynb`](notebooks/pseudo_hessian_overthrust.ipynb) | Overthrust | conventional FWI · iFWI (SIREN) · iFWI **+ pseudo-Hessian** |
| [`notebooks/hash_encoding_marmousi.ipynb`](notebooks/hash_encoding_marmousi.ipynb) | Marmousi | conventional FWI · iFWI (SIREN) · iFWI **+ hash encoding** |
| [`notebooks/multiparameter_marmousi2.ipynb`](notebooks/multiparameter_marmousi2.ipynb) | Marmousi2 | joint $(v_p, Z)$ implicit FWI **imaging** (variable density, hash + multiscale) |

The two single-parameter methods (which the *Geophysics* paper builds on) were first
presented as EAGE extended abstracts, reproduced here:
* *Implicit full waveform inversion with energy-weighted gradient* — DOI [10.3997/2214-4609.202510069](https://doi.org/10.3997/2214-4609.202510069) (pseudo-Hessian / Overthrust)
* *Multiresolution hash encoding for high resolution implicit full waveform inversion* — DOI [10.3997/2214-4609.202510109](https://doi.org/10.3997/2214-4609.202510109) (hash encoding / Marmousi)

The multiparameter Marmousi2 notebook reproduces the joint velocity–impedance imaging
example of the *GJI* paper **Implicit Full Waveform Inversion Imaging**
(DOI [10.1093/gji/ggag277](https://doi.org/10.1093/gji/ggag277)).

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
* **Multiparameter imaging (FWIM)** inverts velocity **and** impedance jointly with a
  single shared hash + SIREN network (`model_i = init_i + std_i · net(coords)[i]`), driven
  by the variable-density `AcousticVRZ` solver. A band-pass multiscale schedule
  (3, 5, 8, all Hz) restarts the network + optimizer at each scale, baking the previous
  scale's model into the init. (Marmousi2 example.) **Reproduction detail:** the zero-phase
  band-pass must be a time-domain `filtfilt` (`torchaudio`, fp64) — a frequency-domain
  `|H|²` multiply leaves a coherent t=0 wrap artefact that destabilises the joint inversion.

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
src/ifwi_sweep.py     solver wiring + SIREN + hash encoding + pseudo-Hessian + VRZ restart + filters
src/models.py         Overthrust/Marmousi vp and Marmousi2 vp+impedance, true/smooth, embedded (zlib+base85)
notebooks/            pseudo_hessian_overthrust + hash_encoding_marmousi + multiparameter_marmousi2
figures/              comparison figures written by the notebooks (and cached results, gitignored)
tools/encode_models.py  regenerates src/models.py from .npy (only needed to update the models)
```

## Installation

A CUDA GPU is required — this repo drives the solver with `impl='c'` (the compiled
CUDA path). Install the **sweep** differentiable solver from PyPI:

```bash
pip install sweepx
```

This provides the `sweep` package used by the code. Full notes are in the
[sweep docs](https://deepwave-kaust.github.io/sweep/getting-started/installation/).

The notebooks additionally need `torch numpy scipy matplotlib jupyter` (all present in a
sweep environment; see [`requirements.txt`](requirements.txt)); the multiparameter notebook
also uses `torchaudio` for the zero-phase band-pass `filtfilt`. No model data files are
needed — the velocity models are embedded in [`src/models.py`](src/models.py).

## Running

With `sweep` installed, run the notebooks:

```bash
cd notebooks
jupyter lab            # open a notebook and run all cells
```

The notebooks ship with `SMOKE = False` (the full paper run — Overthrust 500
iterations, Marmousi 200, Marmousi2 400 = 4 scales × 100). Set `SMOKE = True`
in the config cell for a quick reduced-iteration check (or set `IFWI_EPOCHS`). Inversion results are
cached under `figures/cache_*.npz`, so re-running only re-plots (seconds) — delete
the cache, or change any config value, to recompute. Observed data is generated
on the fly by the same solver, so no pre-computed data is needed.

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

## Citation

If this repository is useful in your research, please cite the relevant paper(s):

**Journal articles**
* S. Wang and T. Alkhalifah, *Implicit Full Waveform Inversion Imaging* — *Geophysical Journal International*, 2026 — DOI [10.1093/gji/ggag277](https://doi.org/10.1093/gji/ggag277) (multiparameter IFWIM / Marmousi2)
* S. Wang and T. Alkhalifah, *Accelerating High Resolution Implicit Full Waveform Inversion* — *Geophysics* (under review) — single-parameter velocity: pseudo-Hessian + hash encoding

**Conference abstracts**
* S. Wang and T. Alkhalifah, *Implicit full waveform inversion with energy-weighted gradient* — EAGE Annual Conference & Exhibition, 2025 — DOI [10.3997/2214-4609.202510069](https://doi.org/10.3997/2214-4609.202510069) (pseudo-Hessian / Overthrust)
* S. Wang, M. Ravasi and T. Alkhalifah, *Multiresolution hash encoding for high resolution implicit full waveform inversion* — EAGE Annual Conference & Exhibition, 2025 — DOI [10.3997/2214-4609.202510109](https://doi.org/10.3997/2214-4609.202510109) (hash encoding / Marmousi)
* S. Wang and T. Alkhalifah, *Accelerating the convergence of implicit FWI and LSRTM with a field data application* — International Meeting for Applied Geoscience & Energy (IMAGE), 2025 — DOI [10.1190/image2025-4302395.1](https://doi.org/10.1190/image2025-4302395.1)

<details>
<summary>BibTeX</summary>

```bibtex
@article{wang2026imaging,
  author  = {Wang, Shaowen and Alkhalifah, Tariq},
  title   = {Implicit Full Waveform Inversion Imaging},
  journal = {Geophysical Journal International},
  year    = {2026},
  doi     = {10.1093/gji/ggag277}
}

@article{wang_accelerating,
  author  = {Wang, Shaowen and Alkhalifah, Tariq},
  title   = {Accelerating High Resolution Implicit Full Waveform Inversion},
  journal = {Geophysics},
  note    = {Under review}
}

@inproceedings{wang2025energyweighted,
  author    = {Wang, Shaowen and Alkhalifah, Tariq},
  title     = {Implicit Full Waveform Inversion with Energy-Weighted Gradient},
  booktitle = {EAGE Annual Conference \& Exhibition},
  year      = {2025},
  doi       = {10.3997/2214-4609.202510069}
}

@inproceedings{wang2025hash,
  author    = {Wang, Shaowen and Ravasi, Matteo and Alkhalifah, Tariq},
  title     = {Multiresolution Hash Encoding for High Resolution Implicit Full Waveform Inversion},
  booktitle = {EAGE Annual Conference \& Exhibition},
  year      = {2025},
  doi       = {10.3997/2214-4609.202510109}
}

@inproceedings{wang2025lsrtm,
  author    = {Wang, Shaowen and Alkhalifah, Tariq},
  title     = {Accelerating the Convergence of Implicit {FWI} and {LSRTM} with a Field Data Application},
  booktitle = {International Meeting for Applied Geoscience \& Energy (IMAGE)},
  year      = {2025},
  doi       = {10.1190/image2025-4302395.1}
}
```
</details>
