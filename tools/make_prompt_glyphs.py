#!/usr/bin/env python3
"""Builds src/engine/prompt_glyphs_data.inc: the Xbox, Steam Deck and Steam
Controller button glyphs engine/prompt_glyphs.cpp writes into a copy of the
player's own menu/common.tpf.dcx.

The images are Kenney's Input Prompts (CC0, tools/prompts/License.txt), not
game files. Unpack the download (kenney.nl/assets/input-prompts) anywhere and
point this at the folder that holds "Xbox Series", "Steam Deck" and "Steam
Controller":

    uv run --with pillow --with numpy --with etcpak tools/make_prompt_glyphs.py --kenney <folder>

For each pack: a BC7 32x32 for each of the twenty KG_* textures and a 96x32
strip for the HUD's Circle, Cross and L3 in MENU_Common_00091, blocks in row
order, with the blocks of each to write and the texture they go over.
"""
import argparse
from pathlib import Path

import etcpak
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'src' / 'engine' / 'prompt_glyphs_data.inc'

# The KG_* textures in kPromptSlots' order (OK is Circle, Cancel Cross).
SLOTS = ('Cancel', 'OK', 'R_L', 'R_U', 'L1', 'L2', 'L3', 'R1', 'R2', 'R3', 'LS', 'RS',
         'L_U', 'L_D', 'L_LR', 'L_UD', 'L_UDLR', 'Start', 'TP_L', 'TP_R')

# Each slot's glyph, by SDL's mapping (SOUTH is Cross; Back / View is the
# touchpad). "face:" a disc with the letter in its own colour, "disc:" and
# "key:" a disc or key cap with the symbol in tan, "pad:" the d-pad with those
# arms lit, "keep" the game's own glyph.
PACKS = {
    'xbox': ('Xbox Series', {
        'Cancel': ['face:xbox_button_color_a'], 'OK': ['face:xbox_button_color_b'],
        'R_L': ['face:xbox_button_color_x'], 'R_U': ['face:xbox_button_color_y'],
        'L1': ['key:xbox_lb'], 'L2': ['key:xbox_lt'], 'L3': ['disc:xbox_ls'],
        'R1': ['key:xbox_rb'], 'R2': ['key:xbox_rt'], 'R3': ['disc:xbox_rs'],
        'LS': ['disc:xbox_stick_top_l'], 'RS': ['disc:xbox_stick_top_r'],
        'L_U': ['pad:u'], 'L_D': ['pad:d'], 'L_LR': ['pad:lr'], 'L_UD': ['pad:ud'], 'L_UDLR': ['pad:udlr'],
        'Start': ['disc:xbox_button_menu'], 'TP_L': ['disc:xbox_button_view'], 'TP_R': ['disc:xbox_button_view'],
    }),
    # The Deck names L1-R3 and its sticks as the DualShock does; its A/B/X/Y are uncoloured.
    'steamdeck': ('Steam Deck', {
        'Cancel': ['disc:steamdeck_button_a'], 'OK': ['disc:steamdeck_button_b'],
        'R_L': ['disc:steamdeck_button_x'], 'R_U': ['disc:steamdeck_button_y'],
        'L1': ['keep'], 'L2': ['keep'], 'L3': ['keep'], 'R1': ['keep'], 'R2': ['keep'], 'R3': ['keep'],
        'LS': ['keep'], 'RS': ['keep'],
        'L_U': ['keep'], 'L_D': ['keep'], 'L_LR': ['keep'], 'L_UD': ['keep'], 'L_UDLR': ['keep'],
        'Start': ['disc:steamdeck_button_options'], 'TP_L': ['disc:steamdeck_button_view'],
        'TP_R': ['disc:steamdeck_button_view'],
    }),
    # One stick (the game's L / L3) and the right trackpad (RP) as the right stick.
    'steamcontroller': ('Steam Controller', {
        'Cancel': ['face:steam_button_color_a'], 'OK': ['face:steam_button_color_b'],
        'R_L': ['face:steam_button_color_x'], 'R_U': ['face:steam_button_color_y'],
        'L1': ['key:steam_lb'], 'L2': ['key:steam_lt'], 'L3': ['keep'],
        'R1': ['key:steam_rb'], 'R2': ['key:steam_rt'], 'R3': ['keep'],
        'LS': ['keep'], 'RS': ['disc:steam_button_rp'],
        'L_U': ['keep'], 'L_D': ['keep'], 'L_LR': ['keep'], 'L_UD': ['keep'], 'L_UDLR': ['keep'],
        'Start': ['disc:steam_button_start_icon'], 'TP_L': ['disc:steam_button_back_icon'],
        'TP_R': ['disc:steam_button_back_icon'],
    }),
}

ICON = 32
# The HUD strip: 24x8 blocks from block (165, 30) of MENU_Common_00091. Its
# three glyphs are 26 pixels wide; HUD_DISCS are their discs' centres.
STRIP_W, STRIP_H = 96, 32
HUD_GLYPH = 26
HUD_GLYPHS = (('OK', 14, 16), ('Cancel', 47, 16), ('L3', 79, 16))
HUD_DISCS = {'OK': (13.5, 15.5), 'Cancel': (46.5, 15.5), 'L3': (78.5, 15.5)}
HUD_COLUMNS = {'OK': range(0, 8), 'Cancel': range(8, 16), 'L3': range(16, 24)}


# ---- The game's style: a dark bevelled cap with only the symbol coloured,
# measured from the 1.09 textures (sizes in pixels of a 32-pixel icon).
CAP_FILL = 46
TAN = np.array([195, 161, 113], dtype=np.float64)
PAD_FILL, PAD_LIT = 63, np.array([125, 101, 55], dtype=np.float64)
SUPER = 8  # drawn 8x, fit() takes it down
CANVAS = 30


def symbol(folder, name):
    """The symbol cut out of a solid Kenney image (its holes, less a ring's
    gap) as a cropped L image, and the image's colour."""
    src = np.asarray(Image.open(folder / 'Default' / f'{name}.png').convert('RGBA'), dtype=np.float64)
    alpha = src[..., 3] / 255.0
    colour = np.median(src[alpha > 0.9][:, :3], axis=0)
    solid = alpha > 0.5
    filled = np.zeros_like(alpha)
    for y, row in enumerate(solid):
        xs = np.nonzero(row)[0]
        if len(xs):
            filled[y, xs[0]:xs[-1] + 1] = 1.0
    holes = np.clip(filled - alpha, 0, 1)
    width = np.ptp(np.nonzero(solid)[1]) + 1
    seen = np.zeros(holes.shape, bool)
    keep = np.zeros(holes.shape, bool)
    h, w = holes.shape
    for y0, x0 in zip(*np.nonzero(holes > 0.5)):
        if seen[y0, x0]:
            continue
        part, todo = [], [(y0, x0)]
        seen[y0, x0] = True
        while todo:
            y, x = todo.pop()
            part.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and holes[ny, nx] > 0.5:
                    seen[ny, nx] = True
                    todo.append((ny, nx))
        ys, xs = zip(*part)
        if len(part) >= 4 and max(xs) - min(xs) < 0.6 * width and max(ys) - min(ys) < 0.6 * width:
            keep[ys, xs] = True
    near = np.asarray(Image.fromarray(keep.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))) > 0
    cut = Image.fromarray((holes * near * 255).astype(np.uint8))
    return cut.crop(cut.getbbox()), colour


def box_sdf(dx, dy, hw, hh, r):
    qx, qy = np.abs(dx) - (hw - r), np.abs(dy) - (hh - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def shape_sdf(kind, dx, dy):
    """Distance outside the cap's edge, in icon pixels (negative inside)."""
    if kind in ('face', 'disc'):
        return np.hypot(dx, dy) - 13
    if kind == 'key':
        return box_sdf(dx, dy, 14.5, 9.5, 3.5)
    if kind == 'pad':
        return np.minimum(box_sdf(dx, dy, 15, PAD_ARM, 1.5), box_sdf(dx, dy, PAD_ARM, 15, 1.5))
    raise ValueError(kind)


PAD_ARM = 4.5
SYMBOL_BOX = {'face': (15, 14), 'disc': (15, 14), 'key': (22, 13)}  # w, h


def put_symbol(rgb, folder, kind, name, centre, scale):
    """The symbol fitted to the kind's box, centred on centre (x, y)."""
    cut, colour = symbol(folder, name)
    colour = colour if kind == 'face' else TAN
    bw, bh = SYMBOL_BOX[kind]
    k = min(bw * scale / cut.width, bh * scale / cut.height)
    w, h = round(cut.width * k), round(cut.height * k)
    mask = np.zeros(rgb.shape[:2])
    top, left = round(centre[1] - h / 2), round(centre[0] - w / 2)
    mask[top:top + h, left:left + w] = np.asarray(cut.resize((w, h), Image.LANCZOS), dtype=np.float64) / 255.0
    return rgb * (1 - mask[..., None]) + colour * mask[..., None]


def dark_glyph(folder, spec, size=(CANVAS, CANVAS), centre=None, finish=False):
    """A whole glyph, cap and all (the d-pad, the HUD's), on a size (w, h)
    canvas drawn SUPER times over; finish scales it down to size."""
    kind, name = spec.split(':', 1)
    cw, ch = size
    cx, cy = centre or (cw / 2, ch / 2)
    yy, xx = np.mgrid[:ch * SUPER, :cw * SUPER] + 0.5
    dx, dy = xx / SUPER - cx, yy / SUPER - cy  # in icon pixels
    d = shape_sdf(kind, dx, dy)
    fill = np.full(dx.shape + (3,), float(CAP_FILL))
    if kind == 'pad':
        lit = {'u': dy < -PAD_ARM, 'd': dy > PAD_ARM, 'l': dx < -PAD_ARM, 'r': dx > PAD_ARM}
        fill[:] = PAD_FILL
        fill[np.any([lit[c] for c in name], axis=0)] = PAD_LIT
    # The bevel: lit at the top left, darker at the bottom right, and a shadow.
    gy, gx = np.gradient(d)
    facing = (gx + gy) / (np.sqrt(2) * np.maximum(np.hypot(gx, gy), 1e-9))  # +1 bottom right
    depth = -d
    lift = 70 * np.clip(-facing, 0, 1) * np.clip((2.3 - depth) / 2, 0, 1)
    sink = 16 * np.clip(facing, 0, 1) * np.clip((2.5 - depth) / 2.5, 0, 1)
    rgb = np.clip(fill + (lift - sink)[..., None], 0, 255)
    if kind != 'pad':
        rgb = put_symbol(rgb, folder, kind, name, (cx * SUPER, cy * SUPER), SUPER)
    cover = np.clip(-d * SUPER, 0, 1)
    shadow = 0.6 * np.clip(0.5 - shape_sdf(kind, dx - 0.7, dy - 1.2), 0, 1)
    alpha = cover + (1 - cover) * shadow
    rgb = rgb * (cover / np.maximum(alpha, 1e-9))[..., None]  # the shadow is black
    image = Image.fromarray(np.concatenate([rgb, alpha[..., None] * 255], axis=2).round().clip(0, 255).astype(np.uint8), 'RGBA')
    return image.convert('RGBa').resize(size, Image.LANCZOS).convert('RGBA') if finish else image


# ---- A disc or key cap writes only the blocks holding its symbol, over the
# game's own cap: every disc over the Square (its symbol fits those blocks), a
# key cap over its own. Mask bit y * 8 + x is block (x, y).
def block_mask(cols, rows):
    return sum(1 << (y * 8 + x) for y in rows for x in cols)


INNER = {'face': block_mask(range(2, 6), range(2, 6)), 'disc': block_mask(range(2, 6), range(2, 6)),
         'key': block_mask(range(1, 7), range(2, 6))}
ALL_BLOCKS = (1 << 64) - 1
DISC_BASE = 'R_L'


def inner_glyph(folder, spec):
    """The symbol on the flat cap colour, 32x32."""
    kind, name = spec.split(':', 1)
    side = ICON * SUPER
    rgb = np.full((side, side, 3), float(CAP_FILL))
    rgb = put_symbol(rgb, folder, kind, name, (side / 2, side / 2), SUPER)
    image = Image.fromarray(np.concatenate([rgb, np.full((side, side, 1), 255.0)], axis=2).round().astype(np.uint8), 'RGBA')
    return image.resize((ICON, ICON), Image.LANCZOS)


def slot_glyph(folder, slot, names):
    """A slot's image, its block mask, and the slot it is written over."""
    if names == ['keep']:
        return Image.new('RGBA', (ICON, ICON)), 0, slot
    kind = names[0].split(':', 1)[0] if len(names) == 1 and ':' in names[0] else None
    if kind in INNER:
        base = DISC_BASE if kind in ('face', 'disc') else slot
        return inner_glyph(folder, names[0]), INNER[kind], base
    return fit(folder, names, ICON), ALL_BLOCKS, slot


def load(folder, name):
    if ':' in name:
        return dark_glyph(folder, name)  # keeps its margin: no crop
    image = Image.open(folder / 'Default' / f'{name}.png').convert('RGBA')
    return image.crop(image.getbbox())


def fit(folder, names, size):
    """The named images in a row, scaled into a size x size square."""
    parts = [load(folder, n) for n in names]
    row = Image.new('RGBA', (sum(p.width for p in parts), max(p.height for p in parts)))
    x = 0
    for p in parts:
        row.alpha_composite(p, (x, (row.height - p.height) // 2))
        x += p.width
    k = (size - 2) / max(row.size)
    w, h = max(1, round(row.width * k)), max(1, round(row.height * k))
    small = row.convert('RGBa').resize((w, h), Image.LANCZOS).convert('RGBA')
    out = Image.new('RGBA', (size, size))
    out.alpha_composite(small, ((size - w) // 2, (size - h) // 2))
    return out


def fill_transparent(image):
    """Transparent pixels take their opaque neighbours' colour, so filtering
    at an edge does not blend in what the PNG kept under alpha 0."""
    px = np.asarray(image, dtype=np.float64)
    rgb, a = px[..., :3], px[..., 3:] / 255.0
    acc, wsum = rgb * a, a.copy()
    for _ in range(8):
        acc = sum(np.roll(np.roll(acc, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        wsum = sum(np.roll(np.roll(wsum, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    spread = np.where(wsum > 0, acc / np.maximum(wsum, 1e-9), 255.0)
    out = np.where(a > 0, rgb, spread)
    return np.concatenate([out, px[..., 3:]], axis=2).round().clip(0, 255).astype(np.int64)


# ---- BC7 by bc7enc, every mode tried: one block can hold cap, label and shadow.
def bc7(image):
    params = etcpak.BC7CompressBlockParams()
    params.init_linear_weights()
    params.m_perceptual = False
    params.m_uber_level = 4
    params.m_max_partitions = 64
    px = fill_transparent(image).astype(np.uint8)
    out = etcpak.compress_bc7(px.tobytes(), image.width, image.height, params)
    assert len(out) == image.width * image.height
    return out


def hud_strip(folder, slots):
    """The strip and the block columns to write (not a "keep" glyph's)."""
    strip = Image.new('RGBA', (STRIP_W, STRIP_H))
    columns = 0
    for slot, cx, cy in HUD_GLYPHS:
        names = slots[slot]
        if names == ['keep']:
            continue
        columns |= sum(1 << c for c in HUD_COLUMNS[slot])
        if len(names) == 1 and names[0].split(':', 1)[0] in ('face', 'disc'):
            strip.alpha_composite(dark_glyph(folder, names[0], (STRIP_W, STRIP_H), HUD_DISCS[slot], finish=True))
        else:
            strip.alpha_composite(fit(folder, names, HUD_GLYPH), (cx - HUD_GLYPH // 2, cy - HUD_GLYPH // 2))
    return strip, columns


def c_array(name, data):
    lines = [f'const std::uint8_t {name}[{len(data)}] = {{']
    for i in range(0, len(data), 24):
        lines.append('    ' + ','.join(f'0x{b:02x}' for b in data[i:i + 24]) + ',')
    lines.append('};')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--kenney', required=True, type=Path, help="the unpacked Input Prompts folder")
    ap.add_argument('--out', type=Path, default=OUT)
    args = ap.parse_args()
    parts = ['// Generated by tools/make_prompt_glyphs.py from Kenney\'s Input Prompts (CC0,',
             '// tools/prompts/License.txt). Do not edit; run the tool again.', '']
    for pack, (folder, slots) in PACKS.items():
        assert set(slots) == set(SLOTS), pack
        folder = args.kenney / folder
        made = [slot_glyph(folder, s, slots[s]) for s in SLOTS]
        icons = b''.join(bc7(image) for image, _, _ in made)
        assert len(icons) == len(SLOTS) * (ICON // 4) ** 2 * 16
        strip_image, columns = hud_strip(folder, slots)
        strip = bc7(strip_image)
        assert len(strip) == (STRIP_W // 4) * (STRIP_H // 4) * 16
        masks = ', '.join(f'0x{m:016x}ull' for _, m, _ in made)
        bases = ', '.join(str(SLOTS.index(b)) for _, _, b in made)
        parts += [c_array(f'kGlyphs_{pack}', icons), c_array(f'kHudStrip_{pack}', strip),
                  f'const std::uint64_t kGlyphMask_{pack}[{len(SLOTS)}] = {{{masks}}};',
                  f'const std::uint8_t kGlyphBase_{pack}[{len(SLOTS)}] = {{{bases}}};',
                  f'const std::uint32_t kHudStripColumns_{pack} = 0x{columns:06x};', '']
        print(f'{pack}: {len(SLOTS)} glyphs and the HUD strip')
    args.out.write_text('\n'.join(parts), encoding='utf-8', newline='\n')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
