# airform

Rhinoplasty outcome simulation from a single photograph. Upload a face, see how
it would look under different surgeons' characteristic results.

**No model is trained anywhere.** A rhinoplasty is seven interpretable numbers.
A surgeon's signature is the mean of those numbers across their public
before/afters. Rendering is a geometric warp. That is the whole system, and it
fits in a weekend.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python opencv-python-headless mediapipe numpy

.venv/bin/python scripts/run_variants.py your-photo.jpg out   # 5 archetypes + contact sheet
.venv/bin/python scripts/verify.py       your-photo.jpg       # locality + closure
.venv/bin/python scripts/make_benchmark.py your-photo.jpg     # ground truth for the bake-off
.venv/bin/python scripts/bench.py                             # score every measurement technique
```

## Why warping instead of image generation

Geometric warping **preserves identity exactly**. Diffusion inpainting of a face
drifts the skin texture, the eye shape, the jawline — and the user says "that
isn't me", which makes the before/after worthless. Warping moves pixels the user
already recognises. It is also deterministic, free, runs in a couple of seconds
on CPU, and supports a continuous before/after slider, which diffusion cannot.

Generative models still have a place here, as a *polish* pass: a low-denoise
img2img masked to the nose, to restore skin texture over stretched pixels. That
is a finishing step, not the renderer.

## Pipeline

| Stage | Module | Contract |
|---|---|---|
| 1. detect | `landmarks.py` | image → `Landmarks` (478 pts, view, yaw) |
| 2. measure | `params.py`, `photometric.py`, `measure/` | `Landmarks` → `NoseParams` |
| 3. fit | `signatures.py` | `[(before, after)]` → `SurgeonSignature` |
| 4. render | `deform.py`, `mls.py` | image + `NoseParams` → image |

Everything crosses module boundaries as the dataclasses in `contracts.py`.
An agent working on one stage needs to read that one file.

`NoseParams` is normalised by interpupillary distance, so it is invariant to
resolution and camera distance. A measurement and a delta are the same type — a
delta is just a difference of measurements — which makes the whole application
one line:

```python
render(image, measure(you).apply(surgeon.delta))
```

Width deltas are stored **proportionally** (−10.7% of your own base), lengths
and angles **absolutely**. A surgeon who takes a 40mm alar base to 36mm performs
"−10%", not "−4mm"; applying −4mm to a 30mm base is a different and much more
aggressive operation. Storing widths absolutely means a signature fitted on one
practice's patient demographic mistransfers to any nose whose baseline differs,
which is a correctness bug and a fairness bug at once, since nasal width varies
systematically with ancestry.

## Rendering: Moving Least Squares

Rigid MLS (Schaefer, McPhail & Warren, SIGGRAPH 2006). 197 control points: 66
nose landmarks that move, 127 face landmarks pinned, 24 border points pinning
the frame. The rigid constraint forbids local shear and scale, which is why
warped skin still reads as skin — the overall narrowing still happens, because
the control-point *set* moves; rigidity only governs how the field interpolates
between them.

Verified:

- **Fidelity** — 100% of requested displacement at control points, 92% between.
- **Identity** — a warp with no displacement is bit-exact (0/255 max error).
- **Locality** — 0.8–5% of pixels change for width edits. Length edits reach
  ~20%, because shortening propagates down the philtrum.
- **Measurements** — alar 0.652, tip 0.365, bridge 0.294, length 0.758 IPD.
  At a 63mm IPD that is 41 / 23 / 18.5 / 48mm, all anatomically normal.

Three implementation details that cost real debugging time:

1. Solve the field **backward** (destination → source). Solving forward and
   inverting leaves holes wherever the warp expands.
2. Upsample the **displacement**, never the absolute coordinate map. `cv2.resize`
   on a coordinate field shifts everything a fraction of a pixel under its
   pixel-centre convention, silently resampling the entire image.
3. Keep the inner loop 2-D. The textbook form implies `(M,N,2)` intermediates
   and is memory-bandwidth bound; collapsing `A_i` to a scaled rotation
   `[[s,t],[-t,s]]` is ~10× faster.

## The finding that matters: FaceMesh cannot measure nose width

MediaPipe's lateral nose vertices come from its shape prior, not the image. The
sidewall is smooth, low-contrast skin with nothing for a landmark regressor to
lock onto, so the prior wins. Warp a nose by a known amount, re-detect, and
almost nothing comes back:

```
adapter      parameter     tone         k       r     rms   verdict
mp_mesh      alar_width    normal   0.063   0.965   12.9%   unusable
mp_mesh      bridge_width  normal  -0.002  -0.178   13.8%   unusable
mp_mesh      tip_width     normal   0.039   0.990   13.3%   unusable
```

`k` is the fraction of a real change the technique reports; `r` is how well it
tracks. Mostly this is severe attenuation rather than pure noise — but a gain of
1/0.06 amplifies noise sixteenfold, so it is unusable either way, and
`bridge_width` has no signal at all.

This does **not** affect rendering, which is verified exact. It affects
*extraction*: signatures fitted from scraped photos would be fitting noise on
the width axes and would render as almost no visible change, with nothing in the
pipeline appearing broken.

`photometric.py` recovers alar width by abandoning landmarks for the measurement
and finding the nasofacial groove — a real shadow edge — with a landmark-seeded
gradient search:

```
photometric  alar_width    normal     0.825   0.999    2.4%   USABLE
photometric  alar_width    dark_flat  0.994   1.000    1.2%   USABLE
photometric  bridge_width  normal     0.500   0.981    8.9%   USABLE
photometric  tip_width     normal     0.089   0.988   12.6%   unusable
```

It is invariant to exposure and contrast for a structural reason: it takes the
*argmax* of gradient magnitude, and argmax does not move under affine intensity
change. Tip width remains unsolved by every technique tried so far.

`scripts/calibrate.py` refuses to emit a gain correction when the response fails
a monotonicity and correlation check, so this class of bug cannot silently
return.

## Measurement bake-off

`scripts/bench.py` scores any measurement technique against ground truth
generated by warping a photo by known amounts (3 parameters × 5 levels × 2 tone
conditions). Adapters implement one small interface (`nosesim/measure/base.py`)
and are compared end to end.

Under evaluation: FaRL/LaPa face-parsing segmentation, MediaPipe Tasks API,
insightface 106-point landmarks, and two refinements of the photometric search.
Results will be committed as they land.

The benchmark's own weakness, stated up front: ground truth comes from warping a
**single subject** with this project's own warp, using MediaPipe-defined bands.
Techniques sharing that nose frame may be flattered, and nothing here shows
generalisation across faces.

## Not built yet

- **Data.** `signatures.ARCHETYPES` holds five hand-authored placeholder deltas
  so the renderer and any frontend have a populated registry to build against.
  `signatures.fit()` is real and ready for scraped pairs today.
- **Profile view.** `NoseParams` carries `dorsal_hump`, `tip_rotation_deg` and
  `tip_projection`; `params.py` computes them and gates them on view. Untested.
  This is the highest-value next step — profile is where rhinoplasty results
  actually show, and the silhouette is an unambiguous edge, so the width
  measurement problem largely disappears.
- **Frontend, API, scraper.**

## Notes

This repository intentionally contains **no face photographs**. Everything
regenerates from your own photo with the commands above.

Surgeons in this codebase are unnamed archetypes by design. Attaching a real
practice's name to a predicted surgical outcome is a defamation surface, and
archetypes communicate the idea better anyway.

Not medical advice, not a surgical planning tool, and not affiliated with any
practice. A geometric estimate is not a prediction of what surgery will do to
your face.
