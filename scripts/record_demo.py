"""Record the README demo: drives the web UI in headless Chromium and saves a GIF.

uv run --with playwright --with imageio-ffmpeg python scripts/record_demo.py
(needs `make up`, and the escalation endpoint reachable for step 4)
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
from playwright.sync_api import sync_playwright

URL = "http://localhost:8000"
OUT = Path("docs/demo.gif")
W, H = 1280, 800


def ask(page, text: str, n: int) -> None:
    page.click("#input")
    page.keyboard.type(text, delay=35)
    page.wait_for_timeout(300)
    page.keyboard.press("Enter")
    page.wait_for_function(
        f"document.querySelectorAll('.msg:not(.user) .meta').length >= {n}", timeout=180000
    )
    page.wait_for_timeout(1400)


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": W, "height": H}, color_scheme="light",
                                  record_video_dir=str(tmp), record_video_size={"width": W, "height": H})  # fmt: skip
        page = ctx.new_page()
        page.goto(URL)
        page.wait_for_timeout(2000)

        ask(page, "Will we run out of Fresh bananas bunch 2 lb before the weekend?", 1)
        page.click(".toggle")  # "Show how": stock, forecast range, reorder math
        page.wait_for_timeout(700)
        page.evaluate("document.getElementById('thread').scrollBy({top: 420, behavior: 'smooth'})")
        page.wait_for_timeout(2600)

        ask(page, "yes, go ahead", 2)
        page.wait_for_timeout(800)
        page.click(".card.inline button[data-act=confirm]")
        page.wait_for_timeout(2200)

        ask(
            page,
            "Which sold more over the last 4 weeks: Ground turkey 1 lb or Chocolate chip cookies 18 oz?",
            3,
        )  # escalates
        page.wait_for_timeout(1200)

        ask(page, "What's the weather tomorrow?", 4)
        page.wait_for_timeout(2000)
        video = page.video.path()
        ctx.close()
        browser.close()

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    palette = tmp / "palette.png"
    vf = "fps=10,scale=960:-1:flags=lanczos"  # ~6.7 MB for a 37 s demo
    subprocess.run([ffmpeg, "-y", "-i", video, "-vf", f"{vf},palettegen=max_colors=128:stats_mode=diff", str(palette)],
                   check=True, capture_output=True)  # fmt: skip
    subprocess.run([ffmpeg, "-y", "-i", video, "-i", str(palette), "-lavfi",
                    f"{vf}[x];[x][1:v]paletteuse=dither=none:diff_mode=rectangle", str(OUT)],
                   check=True, capture_output=True)  # fmt: skip
    print(f"{OUT} {OUT.stat().st_size / 1e6:.1f} MB")
    sys.exit(0)


if __name__ == "__main__":
    main()
