"""Out-of-process FaRL/LaPa face parser.

Runs in .venv-parse (torch 2.2 + numpy<2 + pyfacer), which cannot coexist with
the main venv's numpy 2.x / mediapipe. Communication is deliberately dumb: one
JSON request per stdin line, one JSON reply per stdout line, and the actual
payload goes to disk as a PNG that the caller reads back with its own OpenCV.

What it writes is NOT a binary mask. A hard mask quantises the nose boundary to
whole pixels, and the whole point of this technique is to measure a boundary
position to better than a pixel. So it writes the *decision margin*

    m(x, y) = logit[nose] - max(logit[every other class])

as a uint16 PNG. m > 0 is exactly the argmax nose mask, m = 0 is exactly its
boundary, and because the parser's logits are bilinearly upsampled from the 448
crop, m is smooth enough that the caller can locate the m = 0 crossing by linear
interpolation. Encoding is affine and lossless to ~6e-4 logits:

    stored = (clip(m, -CLIP, CLIP) / CLIP * 0.5 + 0.5) * 65535

Serve mode keeps the 617 MB of weights resident, because the benchmark asks for
32 images and reloading per image would dominate the runtime.
"""

from __future__ import annotations

import json
import os
import sys

MARGIN_CLIP = 20.0  # logits; |m| beyond this carries no positional information
FORMAT_VERSION = 1

_STATE = {}


def _models():
    if not _STATE:
        import facer
        import torch

        torch.set_grad_enabled(False)
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        except Exception:
            pass
        _STATE["torch"] = torch
        _STATE["facer"] = facer
        _STATE["det"] = facer.face_detector("retinaface/mobilenet", device="cpu")
        _STATE["par"] = facer.face_parser("farl/lapa/448", device="cpu")
    return _STATE


def parse_one(image_path: str, out_png: str) -> dict:
    import cv2
    import numpy as np

    st = _models()
    torch, facer = st["torch"], st["facer"]

    img = cv2.imread(image_path)
    if img is None:
        return {"ok": False, "error": f"unreadable image: {image_path}"}

    tens = facer.hwc2bchw(torch.from_numpy(img[:, :, ::-1].copy()))
    with torch.inference_mode():
        faces = st["det"](tens)
        if faces is None or len(faces.get("rects", [])) == 0:
            return {"ok": False, "error": "no face detected"}
        faces = st["par"](tens, faces)

    logits = faces["seg"]["logits"]          # (n_faces, C, H, W)
    names = list(faces["seg"]["label_names"])

    # Largest detected face wins; the bench images are single-subject but a real
    # photo may catch a bystander.
    rects = faces["rects"].cpu().numpy()
    areas = (rects[:, 2] - rects[:, 0]) * (rects[:, 3] - rects[:, 1])
    fi = int(np.argmax(areas))

    # .clone() first: the parser ran under inference_mode, and its output cannot
    # be updated in place outside it.
    lg = logits[fi].float().clone()
    ni = names.index("nose")
    nose = lg[ni].clone()
    lg[ni] = -1e9
    other = lg.max(0).values
    margin = (nose - other).cpu().numpy()

    enc = np.clip(margin, -MARGIN_CLIP, MARGIN_CLIP) / MARGIN_CLIP * 0.5 + 0.5
    enc = np.clip(np.rint(enc * 65535.0), 0, 65535).astype(np.uint16)

    tmp = out_png + ".tmp.png"
    if not cv2.imwrite(tmp, enc):
        return {"ok": False, "error": f"could not write {tmp}"}
    os.replace(tmp, out_png)

    meta = {
        "ok": True,
        "version": FORMAT_VERSION,
        "clip": MARGIN_CLIP,
        "height": int(enc.shape[0]),
        "width": int(enc.shape[1]),
        "labels": names,
        "rect": [float(x) for x in rects[fi]],
        "png": os.path.basename(out_png),
    }
    with open(out_png + ".json", "w") as f:
        json.dump(meta, f)
    return meta


def serve():
    sys.stderr.write("faceparse worker: loading models\n")
    sys.stderr.flush()
    _models()
    sys.stdout.write(json.dumps({"ok": True, "ready": True}) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("cmd") == "quit":
                break
            res = parse_one(req["image"], req["out"])
        except Exception as e:  # never let one bad image kill the server
            res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()


def main():
    if "--serve" in sys.argv:
        serve()
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        sys.stderr.write("usage: _faceparse_worker.py IMAGE OUT.png | --serve\n")
        raise SystemExit(2)
    print(json.dumps(parse_one(args[0], args[1])))


if __name__ == "__main__":
    main()
