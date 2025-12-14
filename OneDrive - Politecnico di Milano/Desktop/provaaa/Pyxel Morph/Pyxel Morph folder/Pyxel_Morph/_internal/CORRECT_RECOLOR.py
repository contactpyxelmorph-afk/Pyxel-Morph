#!/usr/bin/env python3
"""
CORRECT_RECOLOR.py

Tile-based palette pipeline (deterministic, exact-color extraction, subset recycling,
greedy merge-to-limit, priority-color protection, recolor + green preview + verification).

This script is designed to be loaded and called directly by the main application
using the run_recolor_pipeline function.

NOTE: The core logic for subset_recycle and greedy_merge_to_limit is derived
directly from the user's provided algorithms.
"""
from PIL import Image
import numpy as np
import argparse
import json
import csv
import os
import sys
from collections import Counter, defaultdict

# Deterministic seed for any randomness (not used here but kept for reproducibility conceptually)
RNG_SEED = 0

# Default config
DEFAULT_TILE = 8
DEFAULT_MAX_PALETTES = 8

# GB preview green shades (index order 0..3). Keep index mapping consistent.
GB_PALETTE_HEX = ["#E0F8CF", "#86C06C", "#071821", "#306850"]


# ----------------------------
# General Helpers
# ----------------------------
def hex_to_rgb(h):
    h = h.strip()
    if h.startswith('#'):
        h = h[1:]
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_to_hex(rgb):
    r, g, b = rgb
    return "#{:02X}{:02X}{:02X}".format(int(r), int(g), int(b))


def ensure_png_path(path):
    root, ext = os.path.splitext(path)
    if ext == "":
        return root + ".png"
    return path


def rgb_dist_sq(a, b):
    return (int(a[0]) - int(b[0])) ** 2 + (int(a[1]) - int(b[1])) ** 2 + (int(a[2]) - int(b[2])) ** 2


# ----------------------------
# Tile extraction
# ----------------------------
def get_top_colors_from_tile(tile_img, max_colors=4):
    counts = tile_img.getcolors(256)  # FIXED for PyInstaller

    if counts is None:
        data = list(tile_img.getdata())
        freq = Counter(data)
        colors = [c for c, _ in freq.most_common(max_colors)]
    else:
        counts.sort(reverse=True, key=lambda x: x[0])
        colors = [rgb for _, rgb in counts[:max_colors]]

    # pad with black if needed
    while len(colors) < max_colors:
        colors.append((0, 0, 0))

    return colors



def build_tiles(img, tile_size):
    w, h = img.size
    tiles = {}
    for ty in range(0, h, tile_size):
        for tx in range(0, w, tile_size):
            # Ensure crop bounds are within image limits
            crop_right = min(tx + tile_size, w)
            crop_bottom = min(ty + tile_size, h)

            tile = img.crop((tx, ty, crop_right, crop_bottom))
            colors = get_top_colors_from_tile(tile, max_colors=4)
            tiles[(tx, ty)] = {'tile': tile, 'colors': colors}
    return tiles


# ----------------------------
# Palette Optimization (User's Algorithm)
# ----------------------------
def palette_real_colors(p):
    """
    Helper: Return palette colors excluding the black sentinel (0,0,0).
    """
    return [c for c in p if c != (0, 0, 0)]


def subset_recycle(unique_palettes, tile_palette_map):
    """
    Palette Optimization Step 1: Subset Recycling. (User's provided logic)
    """
    changed = True
    while changed:
        changed = False
        n = len(unique_palettes)
        i = 0
        while i < n:
            pi = unique_palettes[i]
            real_pi = palette_real_colors(pi)
            if not real_pi:
                i += 1
                continue
            j = 0
            merged_this_i = False
            while j < n:
                if i == j:
                    j += 1
                    continue
                pj = unique_palettes[j]
                real_pj = palette_real_colors(pj)

                # Check if real_pi is a subset of real_pj
                is_subset = all(c in real_pj for c in real_pi)

                if is_subset:
                    # Reassign tiles using palette i -> j
                    for coord, idx in list(tile_palette_map.items()):
                        if idx == i:
                            tile_palette_map[coord] = j
                    # Remove palette i and update indices > i
                    unique_palettes.pop(i)
                    for coord in tile_palette_map:
                        if tile_palette_map[coord] > i:
                            tile_palette_map[coord] -= 1
                    changed = True
                    merged_this_i = True
                    n -= 1
                    break
                j += 1
            if not merged_this_i:
                i += 1
    return unique_palettes, tile_palette_map


def greedy_merge_to_limit(unique_palettes, tile_palette_map, max_palettes, priority_colors=set(), priority_weight=1.0):
    """
    Palette Optimization Step 2: Greedy Merging to Max Limit. (User's provided logic)
    """

    def usage_counts_from_map(tile_map):
        u = defaultdict(int)
        for idx in tile_map.values():
            u[idx] += 1
        return u

    def centroid(pal):
        # Filter out black sentinel before calculating centroid
        real_colors = np.array([c for c in pal if c != (0, 0, 0)], dtype=float)
        if real_colors.size == 0:
            return np.array([0.0, 0.0, 0.0])
        return real_colors.mean(axis=0)

    pals = [list(p) for p in unique_palettes]
    tile_map = dict(tile_palette_map)

    # loop until at or below limit
    while len(pals) > max_palettes:
        usage = usage_counts_from_map(tile_map)

        # define effective usage applying priority weight (higher weight => less likely to be "least")
        def effective_usage(i):
            pal_has_priority = any(c in priority_colors for c in palette_real_colors(pals[i]))
            if pal_has_priority:
                # Palettes with priority colors are artificially weighted higher so they are selected later
                return usage.get(i, 0) * priority_weight
            return usage.get(i, 0)

        # Find the least used palette index (lowest effective usage)
        least = min(range(len(pals)), key=lambda i: (effective_usage(i), i))
        if len(pals) == 1:
            break

        cents = [centroid(p) for p in pals]

        # find best target for least
        best_target = None
        best_dist = None
        for j in range(len(pals)):
            if j == least:
                continue

            least_real_colors = palette_real_colors(pals[least])
            least_has_priority = any(c in priority_colors for c in least_real_colors)
            target_has_priority = any(c in priority_colors for c in palette_real_colors(pals[j]))

            # Constraint check: Do not merge a priority-containing palette into
            # a non-priority target. This prevents losing the priority color through merging.
            if least_has_priority and not target_has_priority:
                continue

            d = np.sum((cents[least] - cents[j]) ** 2)
            if best_dist is None or d < best_dist:
                best_dist = d
                best_target = j

        # If strict priority constraints leave no target (highly unlikely), fall back to nearest neighbor
        if best_target is None:
            best_target = min([j for j in range(len(pals)) if j != least],
                              key=lambda j: np.sum((cents[least] - cents[j]) ** 2))

        # merge least -> best_target
        merged = list(pals[best_target])
        for c in pals[least]:
            if c not in merged and c != (0, 0, 0):  # Don't re-add black sentinel
                merged.append(c)

        # Pad or truncate the merged palette to exactly 4 colors
        merged = merged[:4]
        while len(merged) < 4:
            merged.append((0, 0, 0))

        pals[best_target] = merged

        # reassign tiles mapped to least -> best_target
        for coord, idx in tile_map.items():
            if idx == least:
                tile_map[coord] = best_target

        # remove least and shift indices
        pals.pop(least)
        new_map = {}
        for coord, idx in tile_map.items():
            if idx > least:
                new_map[coord] = idx - 1
            else:
                new_map[coord] = idx
        tile_map = new_map

    return pals, tile_map


# ----------------------------
# Recolor + green preview generation
# ----------------------------
def generate_recolor_and_preview(img, tiles, tile_map, palettes, tile_size):
    """
    For every tile pixel:
     - determine palette index (tile_map)
     - find slot index in that palette: exact match preferred; else nearest by RGB
     - recolor pixel to palette[slot]
     - preview pixel to GB_PALETTE[slot]
    Returns PIL images recolor_img and preview_img.
    """
    w, h = img.size
    recolor = Image.new("RGB", (w, h))
    preview = Image.new("RGB", (w, h))
    gb_rgbs = [hex_to_rgb(hx) for hx in GB_PALETTE_HEX]
    # Ensure palettes padded to 4 slots (though they should be already)
    pals4 = [list(p[:4]) + [(0, 0, 0)] * (4 - len(p[:4])) for p in palettes]

    for (tx, ty), info in tiles.items():
        pidx = tile_map.get((tx, ty), 0)
        # safety: clamp pidx
        if pidx < 0 or pidx >= len(pals4):
            pidx = 0
        pal = pals4[pidx]

        tile = info['tile']
        tile_w, tile_h = tile.size
        out_tile = Image.new("RGB", (tile_w, tile_h))
        out_tile_preview = Image.new("RGB", (tile_w, tile_h))

        for y in range(tile_h):
            for x in range(tile_w):
                px = tile.getpixel((x, y))
                slot = None

                # 1. Exact match check
                for i, c in enumerate(pal):
                    if px == c:
                        slot = i
                        break

                # 2. Nearest neighbor if no exact match
                if slot is None:
                    best_i = 0
                    # Initialize best distance using the first color (index 0)
                    best_d = rgb_dist_sq(px, pal[0])
                    for i, c in enumerate(pal[1:], start=1):
                        d = rgb_dist_sq(px, c)
                        if d < best_d:
                            best_d = d
                            best_i = i
                    slot = best_i

                slot = max(0, min(3, slot))  # Clamp slot index to 0-3

                mapped = pal[slot]
                out_tile.putpixel((x, y), mapped)
                out_tile_preview.putpixel((x, y), gb_rgbs[slot])

        recolor.paste(out_tile, (tx, ty))
        preview.paste(out_tile_preview, (tx, ty))

    return recolor, preview


# ----------------------------
# Verification
# ----------------------------
def verify_mapping(recolor_img, preview_img, tiles, tile_map, palettes, tile_size):
    """
    Reconstruct recolor image from preview_img and palettes/tile_map, compare with recolor_img.
    Returns (mismatch_count, debug_img_or_None)
    """
    w, h = recolor_img.size
    # Ensure palette map is padded to 4 colors
    pal_map = [list(p[:4]) + [(0, 0, 0)] * (4 - len(p[:4])) for p in palettes]
    gb_rgbs = [hex_to_rgb(hx) for hx in GB_PALETTE_HEX]
    reconstruct = Image.new("RGB", (w, h))

    for (tx, ty), info in tiles.items():
        pidx = tile_map.get((tx, ty), 0)
        if pidx < 0 or pidx >= len(pal_map):
            pidx = 0
        pal = pal_map[pidx]

        tile_prev = preview_img.crop((tx, ty, tx + tile_size, ty + tile_size))
        tile_w, tile_h = tile_prev.size

        out_tile = Image.new("RGB", (tile_w, tile_h))

        for y in range(tile_h):
            for x in range(tile_w):
                prev_px = tile_prev.getpixel((x, y))
                slot = None

                # Map GB shade back to slot index
                for i, g in enumerate(gb_rgbs):
                    if prev_px == g:
                        slot = i
                        break

                if slot is None:
                    # Fallback for nearest slot if exact match failed (e.g., due to rounding issues, though unlikely)
                    best_i = 0
                    best_d = rgb_dist_sq(prev_px, gb_rgbs[0])
                    for i, g in enumerate(gb_rgbs[1:], start=1):
                        d = rgb_dist_sq(prev_px, g)
                        if d < best_d:
                            best_d = d
                            best_i = i
                    slot = best_i

                slot = max(0, min(3, slot))
                out_tile.putpixel((x, y), pal[slot])

        reconstruct.paste(out_tile, (tx, ty))

    a = np.array(recolor_img).astype(np.int16)
    b = np.array(reconstruct).astype(np.int16)
    diff = np.any(a != b, axis=2)
    mismatches = int(diff.sum())

    debug_img = None
    if mismatches > 0:
        # create debug overlay: recolor with semi-transparent red where mismatch
        debug = recolor_img.convert("RGBA")
        overlay = Image.new("RGBA", (w, h), (255, 0, 0, 0))
        ov = np.array(overlay)
        mask = diff

        # Apply semi-transparent red color
        ov[mask, 0] = 255
        ov[mask, 1] = 0
        ov[mask, 2] = 0
        ov[mask, 3] = 140  # Alpha

        overlay = Image.fromarray(ov, mode="RGBA")
        debug_img = Image.alpha_composite(debug, overlay)
    return mismatches, debug_img


# ----------------------------
# Main Wrapper for GUI (The REQUIRED entry point)
# ----------------------------
# CRITICAL: This function must be named exactly 'run_recolor_pipeline'
# as it is the mandatory entry point for the desktop application runner.


def run_recolor_pipeline(input_path, output_prefix, max_palettes, priority_hex="", priority_weight=1.0,
                         tile_size=DEFAULT_TILE):
    """
    Main entry point for the GUI. Takes arguments directly, avoiding argparse.
    Returns True on success, False on known failure.
    """
    try:
        # Cast inputs based on function signature
        max_palettes = int(max_palettes)
        print(f"[RECOLOR] Using max_palettes = {max_palettes}")
        tile_size = int(tile_size)
        priority_weight = float(priority_weight)

        if not os.path.exists(input_path):
            print(f"ERROR: input not found: {input_path}", file=sys.stderr)
            return False

        # parse priority colors
        priority_colors = set()
        if priority_hex:
            for tok in priority_hex.split(','):
                tok = tok.strip()
                if tok:
                    try:
                        priority_colors.add(hex_to_rgb(tok))
                    except Exception:
                        print(f"Warning: invalid priority color ignored: {tok}")

        # load image
        img = Image.open(input_path).convert("RGB")
        w, h = img.size
        if w % tile_size != 0 or h % tile_size != 0:
            print("WARNING: image dimensions not divisible by tile size; partial edge tiles will be included.")

        # 1. Build tiles
        print("Building tiles and extracting top colors...")
        tiles = build_tiles(img, tile_size)

        # 2. Initial exact palette assignment
        print("Initial exact palette assignment...")
        palette_map = {}
        unique_palettes = []
        tile_palette_map = {}

        for coord, info in tiles.items():
            key = tuple(info['colors'])
            if key in palette_map:
                pidx = palette_map[key]
            else:
                pidx = len(unique_palettes)
                unique_palettes.append(list(info['colors']))
                palette_map[key] = pidx
            tile_palette_map[coord] = pidx

        print("Initial palette count:", len(unique_palettes))

        # 3. Subset recycling (post-process)
        print("Running subset recycling...")
        unique_palettes, tile_palette_map = subset_recycle(unique_palettes, tile_palette_map)
        print("Palette count after subset recycling:", len(unique_palettes))

        # 4. Greedy merge to limit while respecting priority colors
        if len(unique_palettes) > max_palettes:
            print(
                f"Merging greedy to limit {max_palettes} (priority colors: {len(priority_colors)}, weight={priority_weight})...")
            unique_palettes, tile_palette_map = greedy_merge_to_limit(unique_palettes, tile_palette_map, max_palettes,
                                                                      priority_colors, priority_weight)
            print("Palette count after greedy merging:", len(unique_palettes))

        # 5. Compact indices and pad palettes
        used = sorted(set(tile_palette_map.values()))
        idx_map = {old: i for i, old in enumerate(used)}
        unique_palettes = [unique_palettes[old] for old in used]
        tile_palette_map = {coord: idx_map[idx] for coord, idx in tile_palette_map.items()}

        # Ensure palette items are padded to length 4
        for i, p in enumerate(unique_palettes):
            while len(p) < 4:
                p.append((0, 0, 0))
            unique_palettes[i] = p[:4]

        # --- HARD CAP: enforce final palette count ---
        valid_indices = set(range(len(unique_palettes)))
        new_map = {}
        for coord, idx in tile_palette_map.items():
            if idx in valid_indices:
                new_map[coord] = idx
            else:
                new_map[coord] = 0  # fallback palette
        tile_palette_map = new_map

        # 6. Save files (JSON, TXT, CSV)
        palettes_hex = [[rgb_to_hex(c) for c in p] for p in unique_palettes]
        palettes_json_path = f"{output_prefix}_palettes.json"
        with open(palettes_json_path, "w", encoding="utf-8") as f:
            json.dump(palettes_hex, f, indent=2)

        palettes_txt_path = f"{output_prefix}_palettes.txt"
        with open(palettes_txt_path, "w", encoding="utf-8") as f:
            for i, p in enumerate(palettes_hex):
                f.write(f"Palette {i}: {p}\n")

        # Save tile->palette CSV
        csv_path = f"{output_prefix}_tile_palettes.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["tile_x", "tile_y", "palette_index"])
            for (tx, ty), pidx in tile_palette_map.items():
                writer.writerow([tx // tile_size, ty // tile_size, pidx])

        # 7. Generate recolor and preview
        recolor_img, preview_img = generate_recolor_and_preview(img, tiles, tile_palette_map, unique_palettes,
                                                                tile_size)
        recolor_path = ensure_png_path(f"{output_prefix}_recolor.png")
        preview_path = ensure_png_path(f"{output_prefix}_green_preview.png")
        recolor_img.save(recolor_path)
        preview_img.save(preview_path)

        # 8. Verification (optional debug output)
        mismatches, debug_img = verify_mapping(recolor_img, preview_img, tiles, tile_palette_map, unique_palettes,
                                               tile_size)
        if mismatches > 0 and debug_img is not None:
            dbg_path = f"{output_prefix}_mismatch_debug.png"
            debug_img.save(dbg_path)

        # Final summary and success
        print(f"DEBUG: Pipeline finished. Final palettes: {len(unique_palettes)}. Mismatches: {mismatches}")
        return True


    except Exception as e:

        import traceback

        err = "CRITICAL RECOLOR PIPELINE ERROR (Internal): {}\n{}".format(

            e,

            traceback.format_exc()

        )

        # Save error for the GUI

        global __last_error__

        __last_error__ = err

        print(err, file=sys.stderr)

        return False


# ----------------------------
# Command Line Entry Point (Kept for external execution)
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Tile-based palette pipeline (exact colors).")
    parser.add_argument("input", help="input image path (png/jpg)")
    parser.add_argument("prefix", help="output prefix")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE, help="tile size (default 8)")
    parser.add_argument("--max-palettes", type=int, default=DEFAULT_MAX_PALETTES, help="max palettes to keep")
    parser.add_argument("--priority", type=str, default="",
                        help='Comma separated hex colors to protect, e.g. "#E0F8CF,#FF0000"')
    parser.add_argument("--priority-weight", type=float, default=1.0,
                        help="multiplier applied to usage of palettes that contain a priority color (default 1.0). Larger -> stronger protection.")
    args = parser.parse_args()

    # Call the main logic wrapper
    run_recolor_pipeline(
        input_path=args.input,
        output_prefix=args.prefix,
        max_palettes=args.max_palettes,
        priority_hex=args.priority,
        priority_weight=args.priority_weight,
        tile_size=args.tile_size
    )


if __name__ == "__main__":
    main()