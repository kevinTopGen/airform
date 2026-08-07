#!/usr/bin/env python3
import re
import sys
import urllib.request

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"


def main() -> None:
    url = sys.argv[1]
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    html = urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "replace")
    imgs = re.findall(r"https?://[^\"'\s>]+\.(?:jpe?g|png|webp)", html, re.I)
    # also relative uploads
    rel = re.findall(r"[\"'](/wp-content/uploads/[^\"']+\.(?:jpe?g|png|webp))[\"']", html, re.I)
    from urllib.parse import urljoin

    all_u = list(dict.fromkeys(imgs + [urljoin(url, r) for r in rel]))
    for u in all_u:
        low = u.lower()
        if any(x in low for x in ("logo", "icon", "sprite", "favicon", "blank.gif")):
            continue
        if "upload" in low or "patient" in low or "rhino" in low or "before" in low or "after" in low or "split" in low:
            print(u)


if __name__ == "__main__":
    main()

