from pathlib import Path
from PIL import Image, ImageOps, ImageChops, ImageStat


def features(im: Image.Image) -> dict:
    g = ImageOps.grayscale(im.convert("RGB"))
    w, h = g.size
    thr = ImageStat.Stat(g).mean[0] - 15
    pix = list(g.getdata())
    xs, weights = [], []
    for y in range(int(h * 0.1), int(h * 0.9)):
        for x in range(int(w * 0.05), int(w * 0.95)):
            v = pix[y * w + x]
            if v < thr:
                xs.append(x)
                weights.append(thr - v + 1)
    if not xs:
        return {"off": 0, "sym": 0, "edge_ratio": 0, "dL": 0, "dR": 0, "dM": 0}
    cx = sum(x * wt for x, wt in zip(xs, weights)) / sum(weights)
    cxn = cx / w
    band = g.crop((0, int(h * 0.2), w, int(h * 0.8)))
    bw, bh = band.size
    sl = max(1, bw // 5)
    dL = 255 - ImageStat.Stat(band.crop((0, 0, sl, bh))).mean[0]
    dR = 255 - ImageStat.Stat(band.crop((bw - sl, 0, bw, bh))).mean[0]
    dM = 255 - ImageStat.Stat(band.crop((bw // 3, 0, 2 * bw // 3, bh))).mean[0]
    L = g.crop((0, 0, w // 2, h))
    R = ImageOps.mirror(g.crop((w // 2, 0, w, h)).resize(L.size))
    sym = ImageStat.Stat(ImageChops.difference(L, R)).mean[0]
    return {
        "off": abs(cxn - 0.5),
        "sym": sym,
        "edge_ratio": max(dL, dR) / (dM + 1e-6),
        "dL": dL,
        "dR": dR,
        "dM": dM,
        "cxn": cxn,
    }


def classify(im: Image.Image) -> str:
    f = features(im)
    off, sym = f["off"], f["sym"]
    # True profile: face mass shifted + poor L/R mirror symmetry
    if off >= 0.10 and sym >= 30:
        return "side"
    if off >= 0.08 and sym >= 45:
        return "side"
    # Strong single-edge silhouette with decent asymmetry
    if f["edge_ratio"] >= 1.05 and sym >= 35 and off >= 0.06:
        return "side"
    # Front: centered, relatively symmetric
    if off <= 0.055 and sym <= 42:
        return "front"
    if off <= 0.04:
        return "front"
    return "oblique"


if __name__ == "__main__":
    paths = sorted(Path("tmp_qc").glob("z_*.jpg")) + sorted(Path("tmp_qc").glob("of10_*.jpg"))
    for p in paths:
        im = Image.open(p)
        f = features(im)
        print(
            f"{p.name:14} {classify(im):8} off={f['off']:.3f} sym={f['sym']:.1f} "
            f"er={f['edge_ratio']:.2f} cxn={f.get('cxn',0):.2f}"
        )

