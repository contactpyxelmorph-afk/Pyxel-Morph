#!/usr/bin/env python3

"""

gb_palette_optimize_feature.py (cleaned)



Usage:

    python gb_palette_optimize_feature.py input.png output_prefix

Outputs:

 - {prefix}_original.png

 - {prefix}_palettes.json

 - {prefix}_tile_palettes.csv

 - {prefix}_green_preview.png

"""


from PIL import Image

import numpy as np

import cv2

from collections import Counter

import json

import csv

import sys

import random

# -------------------------

# Config

# -------------------------

TILE_SIZE = 8

MAX_PALETTES = 8

MAX_ITER = 40

GB_PALETTE_HEX = [

    "#E0F8CF",

    "#86C06C",

    "#071821",

    "#306850"

]


# -------------------------
import importlib.util
import sys
from types import ModuleType

def import_bingo_module(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("bingo_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bingo_module"] = module
    spec.loader.exec_module(module)
    return module

def save_pixel_color_csv(output_path, img):
    """
    Saves the coordinates and color of every pixel, including tile index.
    """
    w, h = img.size
    TILE_SIZE = 8  # Use the constant defined globally in BINGO.py

    with open(output_path, "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["pixel_x", "pixel_y", "tile_x", "tile_y", "color_hex"])

        for py in range(h):
            for px in range(w):
                color_rgb = img.getpixel((px, py))
                color_hex = rgb_to_hex(color_rgb)  # Requires your existing rgb_to_hex helper

                tile_x = px // TILE_SIZE
                tile_y = py // TILE_SIZE

                writer.writerow([px, py, tile_x, tile_y, color_hex])


def save_tile_assignments_csv(output_path, assignments):
    """
    Saves the tile coordinates (in pixels and tiles)
    and their assigned palette index to a CSV file.
    """
    # Ensure TILE_SIZE is available globally or passed here.
    # Since it's a constant in BINGO, accessing the global TILE_SIZE is fine.
    # Assuming TILE_SIZE is 8, as defined earlier in your script.
    global TILE_SIZE

    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 🟢 UPDATED HEADER: Now includes all four coordinate columns
        writer.writerow(['pixel_x', 'pixel_y', 'tile_x', 'tile_y', 'palette_index'])

        # Write data rows: (tx, ty) -> pal_idx
        for (tx, ty), pal_idx in assignments.items():
            # Calculate tile indices from pixel coordinates
            tile_x = tx // TILE_SIZE
            tile_y = ty // TILE_SIZE

            # 🟢 UPDATED DATA ROW: (PixelX, PixelY, TileX, TileY, PaletteIndex)
            # Example: (8, 16, 1, 2, 0)
            writer.writerow([tx, ty, tile_x, tile_y, pal_idx])
def hex_to_rgb(h):
    h = h.lstrip('#')

    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(c):
    return "#{:02X}{:02X}{:02X}".format(*c)


def extract_tile_colors(tile):
    pixels = [tile.getpixel((x, y)) for y in range(TILE_SIZE) for x in range(TILE_SIZE)]

    counts = Counter(pixels).most_common(4)

    return [c[0] for c in counts]


def color_distance_sq(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2))


def compute_harris_weights(img_cv, tile_size=TILE_SIZE):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    dst = cv2.dilate(dst, None)

    thresh = 0.01 * dst.max()

    h, w = gray.shape

    weights = {}

    for ty in range(0, h, tile_size):

        for tx in range(0, w, tile_size):
            sub = dst[ty:ty + tile_size, tx:tx + tile_size]

            count = int(np.count_nonzero(sub > thresh))

            weights[(tx, ty)] = count + 1

    return weights


def compute_sift_keypoints_weights(img_cv, tile_size=TILE_SIZE):
    try:

        sift = cv2.SIFT_create()

    except Exception:

        sift = None

    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    if sift is not None:

        kps = sift.detect(gray, None)

    else:

        orb = cv2.ORB_create(nfeatures=1500)

        kps = orb.detect(gray, None)

    h, w = gray.shape

    weights = {(tx, ty): 1 for ty in range(0, h, tile_size) for tx in range(0, w, tile_size)}

    for kp in kps:

        x, y = kp.pt

        tx = int(x // tile_size) * tile_size

        ty = int(y // tile_size) * tile_size

        if (tx, ty) in weights:
            weights[(tx, ty)] += 1

    return weights


def build_tiles(img):
    w, h = img.size

    tiles = {}

    for ty in range(0, h, TILE_SIZE):

        for tx in range(0, w, TILE_SIZE):
            tile = img.crop((tx, ty, tx + TILE_SIZE, ty + TILE_SIZE))

            colors = extract_tile_colors(tile)

            tiles[(tx, ty)] = {'tile': tile, 'colors': colors}

    return tiles


def tile_to_palette_cost(tile_colors, palette):
    cost = 0.0

    for tc in tile_colors:
        best = min(color_distance_sq(tc, pc) for pc in palette)

        cost += best

    return cost


def recompute_palette(assigned_tiles):
    """
    Recomputes a 4-color palette (centroid) using k-means on all colors
    from tiles assigned to this cluster, using OpenCV's stable k-means.
    """
    allcols = []
    for cols in assigned_tiles:
        allcols.extend(cols)
    if not allcols:
        return [(0, 0, 0)] * 4

    pts = np.array(allcols, dtype=np.float32)
    unique = np.unique(pts, axis=0)
    k = min(4, len(unique))

    # Handle case where there are fewer than 4 unique colors
    if len(unique) <= k:
        out = [tuple(map(int, row)) for row in unique.tolist()]
        while len(out) < 4:
            out.append((0, 0, 0))
        return out[:4]

    # --- Use cv2.kmeans for stability ---
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # The flags KMEANS_PP_CENTERS ensures a good initial seed selection
    ret, labels, centroids = cv2.kmeans(pts, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    # ------------------------------------

    cents = [tuple(map(int, np.clip(c, 0, 255))) for c in centroids.reshape(-1, 3)]
    while len(cents) < 4:
        cents.append((0, 0, 0))

    # Sort the colors by perceived brightness (luma)
    cents.sort(key=lambda col: 0.2126 * col[0] + 0.7152 * col[1] + 0.0722 * col[2])
    return cents[:4]


# Assuming 'tiles' is the dictionary of all tile data
import random  # Ensure this is imported at the top of your script


def weighted_assign_and_update(tiles, weights, palettes):
    assignments = {}
    clusters = {i: [] for i in range(len(palettes))}
    total_cost = 0.0

    for coord, info in tiles.items():
        tile_colors = info['colors']
        w = weights.get(coord, 1.0)
        best_idx = None
        best_cost = None

        for i, pal in enumerate(palettes):
            # The cost must still respect the original pixel comparison constraint!
            c = tile_to_palette_cost(tile_colors, pal) * (1.0 + float(w))
            if best_cost is None or c < best_cost:
                best_cost = c
                best_idx = i

        assignments[coord] = best_idx
        clusters[best_idx].append(info['colors'])
        total_cost += best_cost

    new_palettes = []

    # Get a list of all tile colors for re-seeding empty clusters
    all_tile_colors = [info['colors'] for info in tiles.values()]

    for i in range(len(palettes)):
        if not clusters[i]:
            # --- NEW: Re-seed empty cluster ---
            if all_tile_colors:
                # Pick a random tile's colors to become the new seed palette
                seed_colors = random.choice(all_tile_colors)
                new_pal = recompute_palette([seed_colors])
                # Note: recompute_palette is used here just to standardize the format/sorting
            else:
                new_pal = [(0, 0, 0)] * 4
            # --- END NEW ---
        else:
            new_pal = recompute_palette(clusters[i])

        new_palettes.append(new_pal)

    return new_palettes, assignments, total_cost


def initialize_palettes(tiles, initial_k=None):
    num_tiles = max(1, len(tiles))

    if initial_k is None:
        initial_k = max(4, min(12, max(1, num_tiles // 8), MAX_PALETTES * 2))

    tiles_sorted = sorted(tiles.items(), key=lambda kv: -len(set(kv[1]["colors"])))

    seed_tiles = tiles_sorted[:initial_k]

    palettes = []

    for _, info in seed_tiles:

        cols = info["colors"][:4]

        while len(cols) < 4:
            cols.append((0, 0, 0))

        palettes.append(cols)

    while len(palettes) < initial_k:
        palettes.append(palettes[-1][:])

    return palettes


def unique_palettes(palettes):
    seen = []

    for p in palettes:

        tup = tuple(p)

        if tup not in seen:
            seen.append(tup)

    return seen


import numpy as np


def compute_palette_centroid(pal):
    # Helper function to get the average color (centroid) of a palette
    # Excludes the (0,0,0) sentinel color if present.
    arr = np.array([c for c in pal if c != (0, 0, 0)], dtype=float)
    if arr.size == 0:
        return np.array([0.0, 0.0, 0.0])
    return arr.mean(axis=0)


def run_clustering_with_restarts(tiles, weights, max_palettes=MAX_PALETTES, tries=6):
    # --- COMPREHENSIVE INITIALIZATION ---
    # These variables must be initialized outside the loop to prevent UnboundLocalError
    best_palettes = None
    best_assignments = None

    # Initialize costs and fallbacks outside the loop
    min_cost = float('inf')
    current_cost = float('inf')

    # Initialize assignment variables for fallback returns
    final_assignments = {}
    current_palettes = []
    # --- End Initialization ---

    for attempt in range(tries):
        # 1. Initialization: Start with a safe excess of clusters (e.g., 2x target)
        initial_k = max(max_palettes * 2, 8)  # Uses max_palettes=5 -> initial_k=10
        palettes = initialize_palettes(tiles, initial_k=initial_k)

        # 2. Iterative Update (find stable clusters)
        prev_cost = None
        for _ in range(MAX_ITER):
            palettes, assignments, cost = weighted_assign_and_update(tiles, weights, palettes)
            if prev_cost is not None and abs(prev_cost - cost) < 1e-3:
                break
            prev_cost = cost

        # 3. Final cleanup and cost check
        palettes, assignments, current_cost = weighted_assign_and_update(tiles, weights, palettes)

        # Determine used palettes and remap (this gets rid of empty clusters)
        used_indices = sorted(set(assignments.values()))
        current_palettes = [palettes[i] for i in used_indices]

        # 4. POST-PROCESSING: GREEDY MERGE TO LIMIT (COST-BASED)
        current_total_cost = current_cost  # Base cost before any merging begins

        while len(current_palettes) > max_palettes:
            if len(current_palettes) <= 1: break

            best_merge_cost_increase = float('inf')
            best_target_i, best_source_j = -1, -1

            # 4a. Find the MERGE that causes the LEAST increase in the total weighted cost
            num_pals = len(current_palettes)

            for i in range(num_pals):
                for j in range(i + 1, num_pals):

                    # 1. SIMULATE THE MERGE: Create a temporary list of palettes
                    simulated_palettes = list(current_palettes)
                    target_i, source_j = i, j

                    # Merge colors
                    merged_colors = set(simulated_palettes[target_i]) | set(simulated_palettes[source_j])

                    # Create the new, single, 4-color palette
                    merged_real = sorted([c for c in merged_colors if c != (0, 0, 0)],
                                         key=lambda col: 0.2126 * col[0] + 0.7152 * col[1] + 0.0722 * col[2])
                    merged_pal = merged_real[:4] + [(0, 0, 0)] * (4 - len(merged_real[:4]))

                    simulated_palettes[target_i] = merged_pal
                    simulated_palettes.pop(source_j)

                    # 2. Re-assign and calculate the NEW cost
                    _, _, new_cost = weighted_assign_and_update(tiles, weights, simulated_palettes)

                    # 3. Calculate the increase
                    cost_increase = new_cost - current_total_cost

                    # 4. Select the best (lowest cost increase)
                    if cost_increase < best_merge_cost_increase:
                        best_merge_cost_increase = cost_increase
                        best_target_i, best_source_j = target_i, source_j

            # 4b. EXECUTE THE BEST MERGE
            if best_target_i == -1: break

            final_merged_palettes = list(current_palettes)

            # Re-build and apply the best merged palette
            merged_colors = set(final_merged_palettes[best_target_i]) | set(final_merged_palettes[best_source_j])
            merged_real = sorted([c for c in merged_colors if c != (0, 0, 0)],
                                 key=lambda col: 0.2126 * col[0] + 0.7152 * col[1] + 0.0722 * col[2])
            merged_pal = merged_real[:4] + [(0, 0, 0)] * (4 - len(merged_real[:4]))

            final_merged_palettes[best_target_i] = merged_pal
            final_merged_palettes.pop(best_source_j)

            # 4c. Update state for the next iteration
            current_palettes, final_assignments, current_cost = weighted_assign_and_update(tiles, weights,
                                                                                           final_merged_palettes)
            current_total_cost = current_cost  # Update the base cost for the next merge step

        # 5. Track Best Result (using raw cost for best perceptual fit)
        if current_cost < min_cost:
            min_cost = current_cost
            best_palettes = current_palettes
            best_assignments = final_assignments

    # --- FINAL RESULT HANDLING ---

    if best_palettes is None:
        # Fallback if no valid result was found
        print("Warning: Could not find any stable clusters. Returning last attempt.")
        return current_palettes, final_assignments

        # 1. Final Optimization Step: Run one last assignment/update
    # This ensures maximum fidelity against the chosen 5 palettes.
    print("\nRunning final optimal assignment for best fidelity...")

    final_optimal_palettes, final_optimal_assignments, final_min_cost = \
        weighted_assign_and_update(tiles, weights, best_palettes)

    print(f"[OK] Optimization Complete. Final Palettes: {len(final_optimal_palettes)}. Final Cost: {final_min_cost:.2f}")


    # Return the final, high-fidelity assignment.
    return final_optimal_palettes, final_optimal_assignments


def palettes_to_hex(palettes):
    return [[rgb_to_hex(c) for c in p] for p in palettes]


def generate_green_preview(img, tiles, assignments, palettes_hex, gb_palette_hex):
    w, h = img.size

    preview = Image.new("RGB", (w, h))

    palette_mappings = []

    for pal_hex in palettes_hex:

        mapping = {}

        for i, orig_hex in enumerate(pal_hex):
            mapping[orig_hex] = gb_palette_hex[i] if i < len(gb_palette_hex) else gb_palette_hex[-1]

        palette_mappings.append(mapping)

    for (tx, ty), info in tiles.items():

        pal_idx = assignments.get((tx, ty), 0)

        pal_idx = min(pal_idx, len(palette_mappings) - 1)

        mapping = palette_mappings[pal_idx]

        tile = info['tile']

        out_tile = Image.new("RGB", (TILE_SIZE, TILE_SIZE))

        for y in range(TILE_SIZE):

            for x in range(TILE_SIZE):

                px = tile.getpixel((x, y))

                hex_px = rgb_to_hex(px)

                if hex_px in mapping:

                    gb_hex = mapping[hex_px]

                else:

                    pal_orig_hexes = list(mapping.keys())

                    if pal_orig_hexes:

                        orig_rgbs = [hex_to_rgb(s) for s in pal_orig_hexes]

                        dists = [color_distance_sq(px, rg) for rg in orig_rgbs]

                        pick = pal_orig_hexes[int(np.argmin(dists))]

                        gb_hex = mapping[pick]

                    else:

                        gb_hex = gb_palette_hex[0]

                out_tile.putpixel((x, y), hex_to_rgb(gb_hex))

        preview.paste(out_tile, (tx, ty))

    return preview


def run_bingo_pipeline(input_path, output_prefix, max_palettes=8):
    """
    Runs the BINGO pipeline fully in-memory.
    Returns True on success, False on failure.
    """
    try:
        # Open image
        img = Image.open(input_path).convert("RGB")
        w, h = img.size
        if w % TILE_SIZE != 0 or h % TILE_SIZE != 0:
            print(f"ERROR: image dimensions must be divisible by {TILE_SIZE}")
            return False

        # Compute weights, build tiles, and run clustering (omitted for brevity)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        harris_weights = compute_harris_weights(img_cv)
        sift_weights = compute_sift_keypoints_weights(img_cv)
        weights = {coord: harris_weights.get(coord, 1) + sift_weights.get(coord, 0)
                   for coord in harris_weights}
        tiles = build_tiles(img)
        palettes, assignments = run_clustering_with_restarts(tiles, weights, max_palettes=max_palettes)
        palettes_hex = palettes_to_hex(palettes)

        # -----------------------------------
        # 💡 FIXED: Save outputs (including CSV)
        # -----------------------------------

        # 1. Original Image
        img.save(f"{output_prefix}_original.png")

        # 2. Palettes JSON
        with open(f"{output_prefix}_palettes.json", "w", encoding="utf-8") as f:
            json.dump(palettes_hex, f, indent=2)

        # 3. CRITICAL: Tile Assignments CSV (THIS IS THE MISSING STEP)
        csv_path = f"{output_prefix}_tile_palettes.csv"
        save_tile_assignments_csv(csv_path, assignments)

        # 4. Green preview
        preview = generate_green_preview(img, tiles, assignments, palettes_hex, GB_PALETTE_HEX)
        preview.save(f"{output_prefix}_green_preview.png")

        return True

    except Exception as e:
        import traceback
        print(f"CRITICAL BINGO PIPELINE ERROR: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return False
def main():
    if len(sys.argv) < 3:
        print("Usage: python gb_palette_optimize_feature.py input.png output_prefix [--max-palettes N]")
        return

    input_path = sys.argv[1]
    prefix = sys.argv[2]

    # default
    max_palettes = 8  # fallback if user doesn't provide --max-palettes

    # parse optional --max-palettes argument
    if "--max-palettes" in sys.argv:
        idx = sys.argv.index("--max-palettes")
        if idx + 1 < len(sys.argv):
            try:
                max_palettes = int(sys.argv[idx+1])
            except ValueError:
                print("Invalid value for --max-palettes, using default", max_palettes)

    print(f"Using max_palettes = {max_palettes}")
    img = Image.open(input_path).convert("RGB")

    w, h = img.size

    if w % TILE_SIZE != 0 or h % TILE_SIZE != 0:
        print("ERROR: image dimensions must be divisible by", TILE_SIZE)

        return

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    print("Computing Harris weights...")

    harris_weights = compute_harris_weights(img_cv)

    print("Computing SIFT/ORB keypoint weights...")

    sift_weights = compute_sift_keypoints_weights(img_cv)

    weights = {coord: harris_weights.get(coord, 1) + sift_weights.get(coord, 0) for coord in harris_weights}

    print("Extracting tiles and colors...")

    tiles = build_tiles(img)

    print("Running feature-aware clustering (with restarts if needed)...")

    palettes, assignments = run_clustering_with_restarts(tiles, weights, max_palettes=int(sys.argv[idx+1]), tries=6)

    palettes_hex = palettes_to_hex(palettes)

    orig_out = f"{prefix}_original.png"

    img.save(orig_out)

    print("Saved original copy:", orig_out)

    palettes_file = f"{prefix}_palettes.json"

    with open(palettes_file, "w", encoding="utf-8") as f:

        json.dump(palettes_hex, f, indent=2)

    print("Saved palettes JSON:", palettes_file)

    csv_file = f"{prefix}_tile_palettes.csv"

    csv_file = f"{output_prefix}_tile_palettes.csv"
    print("Saved tile->palette CSV:", csv_file, "(skipped actual write in GUI mode)")

    print("Saved tile->palette CSV:", csv_file)

    print("Creating green preview (visualization only)...")

    preview = generate_green_preview(img, tiles, assignments, palettes_hex, GB_PALETTE_HEX)

    preview_file = f"{prefix}_green_preview.png"

    preview.save(preview_file)

    print("Saved green preview:", preview_file)

    print("\n==== SUMMARY ====")

    print("Palettes produced:", len(palettes_hex))

    for i, pal in enumerate(palettes_hex):
        print(f"Palette {i}: {pal}")

    print("Files:")

    print(" -", orig_out)

    print(" -", palettes_file)

    print(" -", csv_file)

    print(" -", preview_file)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python gb_palette_optimize_feature.py input.png output_prefix [--max-palettes N]")
    else:
        input_path = sys.argv[1]
        output_prefix = sys.argv[2]
        max_palettes = 8
        if "--max-palettes" in sys.argv:
            idx = sys.argv.index("--max-palettes")
            if idx + 1 < len(sys.argv):
                try:
                    max_palettes = int(sys.argv[idx + 1])
                except ValueError:
                    print("Invalid value for --max-palettes, using default", max_palettes)
        run_bingo_pipeline(input_path, output_prefix, max_palettes=max_palettes)

