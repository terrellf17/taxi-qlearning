"""Saves an MP4 of the trained agent. Use --renderer gym for Gymnasium's artwork."""

from __future__ import annotations

import argparse

import imageio.v2 as imageio
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from qlearn.envs import make_env

ACTIONS = ["South", "North", "East", "West", "Pickup", "Drop-off"]
PADS = [(0, 0), (0, 4), (4, 0), (4, 3)]
PAD_NAMES = ["R", "G", "Y", "B"]
PAD_COLORS = [(214, 69, 65), (34, 150, 80), (226, 170, 0), (42, 120, 214)]
MAP = ["|R: | : :G|", "| : | : : |", "| : : : : |", "| | : | : |", "|Y| : |B: |"]
CELL, PAD_PX, CAPTION_H = 96, 24, 84
BG, GRIDC, WALL, INK = (250, 250, 246), (215, 214, 206), (40, 40, 38), (20, 20, 20)


def _font(size, bold=False):
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


F_SMALL, F_MED, F_BIG = _font(15), _font(18, True), _font(30, True)


def draw_state(state: int, decode) -> Image.Image:
    row, col, pas, dest = decode(state)
    W = H = 5 * CELL + 2 * PAD_PX
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    def box(r, c, inset=0):
        x0, y0 = PAD_PX + c * CELL + inset, PAD_PX + r * CELL + inset
        return [x0, y0, x0 + CELL - 2 * inset, y0 + CELL - 2 * inset]

    for i, (r, c) in enumerate(PADS):  # pads
        light = tuple(int(v + (255 - v) * 0.72) for v in PAD_COLORS[i])
        d.rectangle(box(r, c), fill=light)
        d.text((box(r, c)[0] + 6, box(r, c)[1] + 4), PAD_NAMES[i], font=F_MED, fill=PAD_COLORS[i])
    r, c = PADS[dest]  # destination
    d.rectangle(box(r, c, 3), outline=PAD_COLORS[dest], width=6)
    d.text((box(r, c)[2] - 46, box(r, c)[1] + 6), "DEST", font=F_SMALL, fill=PAD_COLORS[dest])
    for k in range(6):
        d.line([PAD_PX, PAD_PX + k * CELL, PAD_PX + 5 * CELL, PAD_PX + k * CELL], fill=GRIDC, width=1)
        d.line([PAD_PX + k * CELL, PAD_PX, PAD_PX + k * CELL, PAD_PX + 5 * CELL], fill=GRIDC, width=1)
    for rr, line in enumerate(MAP):  # walls
        for cc in range(4):
            if line[2 * cc + 2] == "|":
                x = PAD_PX + (cc + 1) * CELL
                d.line([x, PAD_PX + rr * CELL, x, PAD_PX + (rr + 1) * CELL], fill=WALL, width=7)
    d.rectangle([PAD_PX, PAD_PX, PAD_PX + 5 * CELL, PAD_PX + 5 * CELL], outline=WALL, width=7)

    tx = box(row, col, 18)  # taxi
    d.rounded_rectangle(tx, radius=12, fill=(250, 196, 30), outline=INK, width=3)
    d.text(((tx[0] + tx[2]) / 2, tx[1] + 16), "TAXI", font=F_SMALL, fill=INK, anchor="mm")
    if pas == 4:  # passenger in taxi
        cx, cy = (tx[0] + tx[2]) / 2, tx[3] - 20
        d.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=(120, 60, 170), outline=INK, width=2)
    else:  # passenger waiting
        b = box(*PADS[pas])
        cx, cy = b[0] + 20, b[3] - 22
        d.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=(120, 60, 170), outline=INK, width=2)
    return img


def add_caption(frame: Image.Image, lines: list[str]) -> Image.Image:
    out = Image.new("RGB", (frame.width, frame.height + CAPTION_H), (255, 255, 255))
    out.paste(frame, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((16, frame.height + 12), lines[0], font=F_MED, fill=INK)
    for i, t in enumerate(lines[1:]):
        d.text((16, frame.height + 40 + 20 * i), t, font=F_SMALL, fill=(82, 81, 78))
    return out


def even(img: Image.Image) -> np.ndarray:
    """Crop to even size for the video encoder."""
    w, h = img.size
    return np.asarray(img.crop((0, 0, w - w % 2, h - h % 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="results/best_q_table.npy")
    ap.add_argument("--out", default="results/trained_agent.mp4")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20_000, help="first episode seed (test seeds)")
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--renderer", choices=["builtin", "gym"], default="builtin")
    a = ap.parse_args()

    Q = np.load(a.q)
    env = make_env(noise_p=a.noise, seed=a.seed, render_mode="rgb_array" if a.renderer == "gym" else None)
    decode = env.unwrapped.decode

    def frame(s):
        if a.renderer == "gym":
            return Image.fromarray(env.render())
        return draw_state(s, lambda x: tuple(decode(x)))

    frames, totals = [], []
    for ep in range(a.episodes):
        s, _ = env.reset(seed=a.seed + ep)
        total, step, done = 0, 0, False
        head = f"Episode {ep + 1}/{a.episodes}   ·   greedy Q-learning policy (ε = 0)"
        for _ in range(a.fps):
            frames.append(even(add_caption(frame(s), [head, "start", ""])))
        while not done:
            act = int(np.argmax(Q[s]))
            s, r, term, trunc, _ = env.step(act)
            total += r
            step += 1
            done = term or trunc
            info = f"step {step}   action: {ACTIONS[act]}   reward {r:+d}   episode total {total:+d}"
            note = "passenger delivered" if term else ("time limit reached" if trunc else "")
            frames.append(even(add_caption(frame(s), [head, info, note])))
        frames += [frames[-1]] * a.fps
        totals.append(total)
    env.close()

    imageio.mimsave(a.out, frames, fps=a.fps, codec="libx264", quality=8, macro_block_size=1)
    print(f"wrote {a.out}: {a.episodes} episodes, rewards {totals}, {len(frames)} frames")


if __name__ == "__main__":
    main()
