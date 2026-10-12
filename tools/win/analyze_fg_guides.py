"""Inspect raw captured depth and motion guides alongside FG outputs and manifest."""
import argparse
import json
import struct
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze FG guides diagnostic capture dump.")
    parser.add_argument("directory", type=Path, help="Directory containing capture files (e.g. depth.bin, motion.bin, guides_manifest.json)")
    parser.add_argument("--motion-threshold", type=float,
                        help="Count displacement above this many render pixels (diagnostic only; default half width)")
    args = parser.parse_args()
    if args.motion_threshold is not None and (not math.isfinite(args.motion_threshold) or args.motion_threshold <= 0):
        parser.error("--motion-threshold must be finite and positive")
    return args


def main():
    args = parse_args()
    dir_path = args.directory
    manifest_path = dir_path / "guides_manifest.json"
    if not manifest_path.is_file():
        print(f"Error: manifest not found at {manifest_path}")
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    w = manifest.get("width", 0)
    h = manifest.get("height", 0)
    if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
        raise ValueError("invalid guide dimensions")
    if manifest.get("depth_format") != 100 or manifest.get("motion_format") != 83:
        raise ValueError("expected VK_FORMAT_R32_SFLOAT and VK_FORMAT_R16G16_SFLOAT")
    failed = False
    render_id = manifest.get("render_frame_index", 0)
    frame_id = manifest.get("frame_index", 0)
    reflex_id = manifest.get("reflex_frame_id", 0)
    print(f"Guide Dump Analysis:")
    print(f"  Frame IDs: render={render_id}, guide={frame_id}, reflex={reflex_id}")
    print(f"  Dimensions: {w}x{h}")
    print(f"  Jitter: {manifest.get('jitter')}")
    print(f"  Delta time ms: {manifest.get('frame_ms')}")
    camera = manifest.get("camera", {})
    print(f"  Camera near={camera.get('near_plane')}, far={camera.get('far_plane')}, vfov={camera.get('vertical_fov')}")

    def raw(name):
        path = dir_path / name
        data = path.read_bytes()
        expected = w * h * 4
        if len(data) != expected:
            raise ValueError(f"{name} size mismatch: got {len(data)}, expected {expected}")
        return data

    low, high = math.inf, -math.inf
    nonfinite = outside = 0
    for (value,) in struct.iter_unpack("<f", raw("depth.bin")):
        if not math.isfinite(value):
            nonfinite += 1
            continue
        low, high = min(low, value), max(high, value)
        outside += not -1e-6 <= value <= 1 + 1e-6
    print(f"  Depth (R32): min={low:.6f}, max={high:.6f}, non-finite count={nonfinite}")
    print(f"  Depth values outside 0..1: {outside}")
    failed |= bool(nonfinite or outside)

    xmin = ymin = math.inf
    xmax = ymax = -math.inf
    nonfinite = large = zero = 0
    # A diagnostic count only, never a validity rule or runtime clamp.
    # Half-frame-width displacement is useful for locating exceptional regions.
    threshold = args.motion_threshold if args.motion_threshold is not None else w * 0.5
    box = [w, h, -1, -1]
    for index, (x, y) in enumerate(struct.iter_unpack("<ee", raw("motion.bin"))):
        nonfinite += int(not math.isfinite(x)) + int(not math.isfinite(y))
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        xmin, xmax = min(xmin, x), max(xmax, x)
        ymin, ymax = min(ymin, y), max(ymax, y)
        zero += x == 0 and y == 0
        if math.hypot(x, y) > threshold:
            large += 1
            px, py = index % w, index // w
            box = [min(box[0], px), min(box[1], py), max(box[2], px), max(box[3], py)]
    print(f"  Motion (RG16): vx range=[{xmin:.4f}, {xmax:.4f}], vy range=[{ymin:.4f}, {ymax:.4f}]")
    print(f"  Motion non-finite component count: {nonfinite}")
    print(f"  Zero motion: {zero}/{w*h} pixels ({100*zero/(w*h):.4f}%)")
    print(f"  Magnitude > {threshold:g} render pixels: {large}/{w*h} ({100*large/(w*h):.4f}%), box={box if large else None}")
    failed |= bool(nonfinite)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
