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

Seven techniques, scored against warps of known size (`scripts/bench.py`).
`k` is the fraction of a real change a technique reports; `r` is how well it
tracks. Adapters implement one small interface (`nosesim/measure/base.py`).

```
                  alar_width      bridge_width    tip_width
photometric_v2    k=0.878 r=.999  k=0.466 r=.997  (withdrawn, see below)
faceparse_hybrid  k=0.876 r=.999  k=0.421 r=.987
photometric       k=0.825 r=.999  k=0.500 r=.981
faceparse         k=0.542 r=.999  k=0.063 r=.987
insightface_106   k=0.236 r=.999  k=0.019 r=.985
mp_tasks          k=0.128 r=.995  k=0.005 r=.357
mp_mesh           k=0.063 r=.965  k=-0.002 r=-.178
```

**Every landmark regressor fails on width.** Three vendors and two
architectures — MediaPipe's legacy mesh, its Tasks API, and insightface — all
land at k=0.01–0.24 on parameters the renderer provably delivers at exactly
1.0000. This is not a MediaPipe bug; it is what regressing to a canonical mesh
does when the image gives you nothing to override the prior with.

**Ship `photometric_v2`.** Use `alar_width`, treat `bridge_width` as advisory,
do not use `tip_width`. The real justification is not `k` — it is that
`photometric_v2` is the only adapter of the seven with a diagonally dominant
3×3 response matrix, which is what signature fitting actually depends on.

### What an adversarial pass overturned

Worth reading before trusting any number above.

**`tip_width` ground truth was fabricated.** Feeding `deform.plan`'s own
control-point displacements back through `params.measure` — no pixels, no
detector — gives `k_true = 1.0000` for alar and bridge, but tip asks −20% and
receives **+2.92%**, k_true=0.132, wrong sign. `params.py` gates the tip on a
`b_limit` that scales with alar width, and `band_width` takes max−min over a
point set whose membership changes as the warp pushes points across that
threshold. Every tip number correlated against a label the renderer never
delivered, so `photometric_v2`'s apparent k=0.596 is withdrawn — its gauge
reads the internostril span, not the lobule.

**`k` is partly a coordinate, not a score.** It is set by where an adapter
places its ruler crossed with the renderer's non-uniform smoothstep blend.
Predicting `k` from edge radius alone reproduces the published values across
five cells without reference to detector quality at all.

**The `dark_flat` condition is not lighting.** It is exactly
`0.4502 × normal + 8.905` (R²=0.99994) — a global affine map, and precisely the
transform an argmax-of-gradient method is constructed to be invariant to.
Effective n is 15, not 30, and there is no lighting evidence here.

**Three inputs return confident, in-range, wrong numbers**, none of them
represented in the benchmark: eyeglasses (bridge **+50.7%**), nose-contouring
makeup (tip +15% from shading alone — the method contours the dorsal light
reflex and contouring paints a fake one), and flat frontal light (bridge
+104%). `alar_width` survives all three, which is a further reason to ship it
first and hold the rest as advisory.

Everything here is one subject, one pose, synthetic warps. `k` shifts ≥10% for
the best adapter when only the nose shape changes on the same photograph, and
the alar winner's margin over second place is 0.2%. Treat every `k` as needing
per-subject recalibration, not as a fixed gain.

### Known defects

- `tip_width` ground truth must be fixed — freeze the lobule index set on the
  baseline instead of re-selecting per level — before any tip result is
  meaningful.
- The module-global FaceMesh in `landmarks.py` has no lock. At 4-way
  concurrency 26/40 detections came back computed from another thread's image.
  Must be thread-local or serialised before serving concurrent requests.
- `bench.py` prints neither the off-diagonal response matrix nor separate
  positive/negative gains. Both are ~10 lines and either would have caught the
  tip problem.

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

## Community preference tournament

The `agent/tournament` branch adds a detachable React/Vite pairwise-comparison
surface under `src/tournament`. It uses the same stable surgeon identity as the
simulation core:

```text
SurgeryCase.surgeonId == Surgeon.id == SurgeonSignature.id
```

For the local fictional-fixture demo:

```bash
npm install
npm run dev
```

Run the tournament checks with `npm test` and `npm run build`. The exact
provider boundary, proposed HTTP routes, scraper-manifest mapping, and merge
procedure are documented in `docs/AIRFORM_INTEGRATION.md` and
`docs/TOURNAMENT_INTEGRATION_SPEC.md`.

The preference score is not a clinical-quality or surgical-safety score.
