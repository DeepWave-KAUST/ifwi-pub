"""Encode .npy velocity models into ``src/models.py`` (zlib-9 + base85, float32).

Same packing scheme as sweep's datasets, but stored as **float32** (lossless), so
the embedded models are bit-exact copies of the source arrays. The shipped repo
needs no .npy files -- this script is only for regenerating ``src/models.py`` from
source arrays. Point MODELS at your .npy files and run:

    python tools/encode_models.py
"""
import base64
import os
import zlib

import numpy as np

# (model, name) -> path to a 2D float .npy
MODELS = {
    ("overthrust", "true"):   "overthrust_true.npy",
    ("overthrust", "smooth"): "overthrust_smooth.npy",
    ("marmousi", "true"):     "marmousi_true.npy",
    ("marmousi", "smooth"):   "marmousi_smooth.npy",
}
OUT = os.path.join(os.path.dirname(__file__), "..", "src", "models.py")

_HEADER = '''"""Embedded Overthrust & Marmousi velocity models for ifwi-pub.

Same scheme as sweep's datasets (zlib-9 + base85 packed into a .py, no external
files), but stored as **float32** so the models are bit-exact (lossless) and the
reproduction is unchanged. Use overthrust(name) / marmousi(name) -> float32 array;
name is "true" or "smooth".
"""
import base64
import zlib

import numpy as np

_MODELS = {
'''

_FOOTER = '''}

def _load(model, name):
    shape, blob = _MODELS[(model, name)]
    raw = zlib.decompress(base64.b85decode("".join(blob)))
    return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()

def overthrust(name="true"):
    """Overthrust vp (187, 401), float32. name: 'true' | 'smooth'."""
    return _load("overthrust", name)

def marmousi(name="true"):
    """Marmousi vp (141, 341), float32. name: 'true' | 'smooth'."""
    return _load("marmousi", name)
'''


def _chunks(s, w=78):
    return [s[i:i + w] for i in range(0, len(s), w)]


def main():
    entries = []
    for (model, name), path in MODELS.items():
        a = np.load(path).astype(np.float32)
        b85 = base64.b85encode(zlib.compress(a.tobytes(), 9)).decode("ascii")
        assert np.array_equal(
            np.frombuffer(zlib.decompress(base64.b85decode(b85)),
                          dtype=np.float32).reshape(a.shape), a), "round-trip mismatch"
        blob = "(\n" + "".join(f"        {c!r}\n" for c in _chunks(b85)) + "    )"
        entries.append(f'    ("{model}", "{name}"): ({tuple(a.shape)}, {blob}),')
        print(f"{model}/{name}: {a.shape}  {len(b85)} base85 chars")
    with open(OUT, "w") as f:
        f.write(_HEADER + "\n".join(entries) + "\n" + _FOOTER)
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
