"""Self-rendered bathymetry basemap tiles.

CARTO started watermarking key-less raster tiles (2026-08), and every
key-free alternative bakes labels, roads or land relief into the image. So
the dashboard draws its own: NOAA NCEI's DEM_global_mosaic (coastal relief
models over an ETOPO base; DEM_all is coastal-only and returns sparse,
wrong-valued blocks at zoom ≤ 6) supplies raw float32 elevation for each
tile's bbox, land is painted
flat and the sea is shaded by depth in the theme palette. Nothing else is
drawn. Rendered PNGs are cached on disk for good under
.cache/bathy_tiles/<STYLE>/ — bump STYLE for any palette/ramp change so
browsers (immutable cache headers) and the disk cache both roll over.
"""
import math
import os
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from cache import record_api_calls, _DISK_CACHE_DIR

STYLE = "v6"                 # v6: retina coast tiles emitted at the full 512 px render
TILE = 256
ZMIN, ZMAX = 5, 13            # 13 = maxZoom 12 + detectRetina
# Render envelope (lon/lat); tiles wholly outside 404 so this isn't an open proxy.
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = -86.0, -55.0, 28.0, 50.0
NOAA_URL = ("https://gis.ngdc.noaa.gov/arcgis/rest/services/"
            "DEM_mosaics/DEM_global_mosaic/ImageServer/exportImage")
TILE_DIR = os.path.join(_DISK_CACHE_DIR, "bathy_tiles", STYLE)

# Depth ramp: the shelf (0–SHELF m) gets half the ramp so nearshore bathymetry
# reads, the slope/canyons (SHELF–DEEP m) get the other half.
SHELF, DEEP = 200.0, 4000.0
PALETTES = {
    "dark":  {"land": (0x2e, 0x2e, 0x3c), "shallow": (0x12, 0x12, 0x19), "deep": (0x05, 0x05, 0x0a)},
    "light": {"land": (0xf7, 0xf7, 0xf9), "shallow": (0xdc, 0xe1, 0xea), "deep": (0xb0, 0xbb, 0xcd)},
}
# Coastline-only styles (transparent except an anti-aliased line on the
# land/sea boundary) drawn ABOVE the wind colour field so the coast stays a
# clean edge while wind shows over land and sea alike. `dpr` picks the line
# weight: with detectRetina a tile pixel is half a CSS pixel.
COAST = {
    "coast-dark":  {"rgb": (0x06, 0x06, 0x0c), "alpha": 0.82, "wash": (0x9c, 0x9c, 0xac), "wash_alpha": 0.26},
    "coast-light": {"rgb": (0x14, 0x14, 0x2a), "alpha": 0.78, "wash": (0xff, 0xff, 0xff), "wash_alpha": 0.36},
}
# `wash` is a translucent land fill drawn with the line: land reads muted and
# (as in the v1 basemap) lighter than the sea while the wind still shows
# through it. Line half-widths are px at the 2× render; the dpr-2 variant is
# displayed at half size, so it draws thicker to land at ~1 CSS px.
COAST_HALF_WIDTH = {1: 2.6, 2: 2.8}   # dpr 2 is emitted at the 2× render itself (512 px), so no doubling
COAST_EDGE = 2.6                       # steeper ramp → a crisp core, anti-aliased only by the 2× → 1× box filter
STYLES = list(PALETTES) + list(COAST)
MARGIN = 32                            # px of neighbour context at the 2× render (seam-free cleanup)
# A connected component may only be filled/dropped when it is small enough
# to lie wholly inside every neighbour's margin window too — otherwise the
# tile that sees all of it removes it and the tile that sees part of it
# keeps it, and the seam shows. 600 px ≈ a 28 px disc, under MARGIN.
MAX_COMPONENT_PX = 600
BLOCK = 2                              # tiles per side fetched from NOAA in one request (2×2 → 4 tiles)

_R = 6378137.0
_ORIGIN = math.pi * _R


def _tile_bbox(z, x, y):
    size = 2 * _ORIGIN / (2 ** z)
    xmin = -_ORIGIN + x * size
    ymax = _ORIGIN - y * size
    return xmin, ymax - size, xmin + size, ymax


def _in_envelope(z, x, y):
    xmin, ymin, xmax, ymax = _tile_bbox(z, x, y)
    lon0, lon1 = xmin / _ORIGIN * 180, xmax / _ORIGIN * 180
    lat0 = math.degrees(math.atan(math.sinh(ymin / _R)))
    lat1 = math.degrees(math.atan(math.sinh(ymax / _R)))
    return lon1 > LON_MIN and lon0 < LON_MAX and lat1 > LAT_MIN and lat0 < LAT_MAX


def _lonlat_to_tile(lon, lat, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


# ── minimal TIFF (uncompressed float32) reader — Pillow isn't a dependency ──
_TIFF_TYPES = {1: ("B", 1), 3: ("H", 2), 4: ("I", 4), 11: ("f", 4), 12: ("d", 8), 16: ("Q", 8)}


def _parse_tiff_f32(buf):
    bo = {b"II": "<", b"MM": ">"}[buf[:2]]
    if struct.unpack(bo + "H", buf[2:4])[0] != 42:
        raise ValueError("not a classic TIFF")
    ifd = struct.unpack(bo + "I", buf[4:8])[0]
    n = struct.unpack(bo + "H", buf[ifd:ifd + 2])[0]
    tags = {}
    for i in range(n):
        off = ifd + 2 + i * 12
        tag, typ, cnt = struct.unpack(bo + "HHI", buf[off:off + 8])
        if typ not in _TIFF_TYPES:
            continue
        fmt, sz = _TIFF_TYPES[typ]
        total = sz * cnt
        if total <= 4:
            data = buf[off + 8:off + 8 + total]
        else:
            p = struct.unpack(bo + "I", buf[off + 8:off + 12])[0]
            data = buf[p:p + total]
        tags[tag] = struct.unpack(bo + fmt * cnt, data)
    w, h = tags[256][0], tags[257][0]
    if tags.get(259, (1,))[0] != 1 or tags.get(258, (32,))[0] != 32 or tags.get(339, (3,))[0] != 3:
        raise ValueError("unexpected TIFF encoding (need uncompressed float32)")
    dt = np.dtype(bo + "f4")
    if 273 in tags:                      # strips
        raw = b"".join(buf[o:o + c] for o, c in zip(tags[273], tags[279]))
        return np.frombuffer(raw, dtype=dt, count=w * h).reshape(h, w)
    # tiled layout; a zero byte count is a sparse (absent) block → NaN
    tw, th = tags[322][0], tags[323][0]
    out = np.full((h, w), np.nan, dtype=np.float32)
    per_row = math.ceil(w / tw)
    for i, (o, c) in enumerate(zip(tags[324], tags[325])):
        if c == 0:
            continue
        t = np.frombuffer(buf[o:o + c], dtype=dt, count=tw * th).reshape(th, tw)
        r, col = (i // per_row) * th, (i % per_row) * tw
        out[r:r + th, col:col + tw] = t[:min(th, h - r), :min(tw, w - col)]
    return out


def _fetch_elev(bbox, px, margin=0):
    """Float32 elevation for the tile bbox, optionally with `margin` extra
    pixels of context on every side (the bbox is widened to match)."""
    if margin:
        xmin, ymin, xmax, ymax = bbox
        m = (xmax - xmin) * margin / px
        bbox, px = (xmin - m, ymin - m, xmax + m, ymax + m), px + 2 * margin
    params = {
        "bbox": ",".join(f"{v:.3f}" for v in bbox), "bboxSR": 3857, "imageSR": 3857,
        "size": f"{px},{px}", "format": "tiff", "pixelType": "F32",
        "interpolation": "RSP_BilinearInterpolation", "f": "image",
    }
    for attempt in (1, 2, 3):
        try:
            r = requests.get(NOAA_URL, params=params, timeout=60)
            r.raise_for_status()
            record_api_calls("bathy_tiles", 1)
            elev = _parse_tiff_f32(r.content)
            break
        except Exception:
            if attempt == 3:
                raise
            time.sleep(0.8 * attempt)
    if elev.shape != (px, px):
        raise ValueError(f"NOAA returned {elev.shape}, wanted {(px, px)}")
    return elev


def _png(rgb):
    h, w, _ = rgb.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rgb.reshape(h, w * 3)], axis=1).tobytes()

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def _png_rgba(rgba):
    h, w, _ = rgba.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rgba.reshape(h, w * 4)], axis=1).tobytes()

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6))
            + chunk(b"IEND", b""))


def _clean(elev):
    e = np.nan_to_num(elev.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    e[e < -20000] = 0.0                  # nodata sentinel
    return e


def _box(a, r):
    """Separable box mean, window 2r+1, edges replicated."""
    r = int(r)
    p = np.pad(a, r, mode="edge")
    c = np.cumsum(p, axis=0)
    c = np.concatenate([np.zeros((1, c.shape[1]), c.dtype), c], axis=0)
    a = (c[2 * r + 1:] - c[:-2 * r - 1]) / (2 * r + 1)
    c = np.cumsum(a, axis=1)
    c = np.concatenate([np.zeros((c.shape[0], 1), c.dtype), c], axis=1)
    return (c[:, 2 * r + 1:] - c[:, :-2 * r - 1]) / (2 * r + 1)


def _coast_mask(elev, z):
    """Land mask for the coastline: the raw 0 m contour turns tidal marsh and
    coastal ponds into a tangle at zoom ≥ 10, so land specks / hairlines
    narrower than ~40 m are opened away, water channels narrower than ~120 m
    are closed (the bays and inlets stay, the marsh creeks go), enclosed
    ponds under ~0.25 km² are filled and islands under ~1.5 km² (never less
    than 0.1 % of the tile) are dropped — all in metres, so every zoom gets
    the same geography: a rule finer than a pixel is skipped rather than
    rounded up (Block Island would otherwise vanish at zoom 7). The
    elevation carries MARGIN px of neighbour context so the cleanup sees
    across tile seams; callers crop afterwards. Pond filling is by size, not
    by edge contact, so a pond straddling a seam is treated alike on both
    sides."""
    from scipy import ndimage
    land = _clean(elev) >= 0
    m_per_px = 156543.03 * 0.77 / (2 ** z) / 2          # 2× render, cos(40°) for this envelope
    m2_per_px = m_per_px ** 2
    struct = np.ones((3, 3), bool)
    it = int(20 / m_per_px)                              # floor: a rule under one pixel is skipped, not rounded up
    if it:
        land = ndimage.binary_opening(land, struct, iterations=min(it, 4))
    it = int(60 / m_per_px)
    if it:
        land = ndimage.binary_closing(land, struct, iterations=min(it, 6))
    hole_px = int(np.clip(2.5e5 / m2_per_px, 4, MAX_COMPONENT_PX))
    speck_px = int(np.clip(1.5e6 / m2_per_px, 4, MAX_COMPONENT_PX))
    for inv in (True, False):                      # water holes, then land specks
        a = ~land if inv else land
        lab, n = ndimage.label(a)
        if not n:
            continue
        edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        sizes = np.bincount(lab.ravel())
        small = sizes < (hole_px if inv else speck_px)
        small[edge] = False
        small[0] = False
        if inv:
            land = land | small[lab]
        else:
            land = land & ~small[lab]
    return land


def _render_coast(mask, style, dpr, z=0):
    """The land wash plus a line along the cleaned land/sea boundary: the
    land mask box-blurred gives a ramp across the coast, 1 − |2·blur − 1|
    peaks on the boundary, and the 2× → 1× box filter anti-aliases both like
    the bathymetry's coast. Composited premultiplied so the line sits on
    the wash. The same weight at every zoom — the mask cleanup is what
    keeps zoom ≥ 10 quiet."""
    c = COAST[style]
    land = mask.astype(np.float32)
    ramp = _box(land, COAST_HALF_WIDTH[dpr])
    line = np.clip((1.0 - np.abs(2.0 * ramp - 1.0)) * COAST_EDGE, 0.0, 1.0)
    line[(ramp < 1e-3) | (ramp > 1 - 1e-3)] = 0.0
    crop = (slice(MARGIN, -MARGIN), slice(MARGIN, -MARGIN))
    la = line[crop] * c["alpha"]
    wa = land[crop] * c["wash_alpha"]
    a = la + wa * (1.0 - la)
    pre = (np.array(c["rgb"], np.float32) * la[..., None]
           + np.array(c["wash"], np.float32) * (wa * (1.0 - la))[..., None])
    h, w = a.shape
    if dpr == 2:
        # the retina variant ships the 2× render itself: a 512 px tile Leaflet
        # shows at 128 CSS px, so a DPR-2 screen is 1:1 and a DPR-3 phone
        # downsamples instead of blurring a 256 px tile up
        a2, pre2 = a, pre
    else:
        a2 = a.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
        pre2 = pre.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
    rgba = np.zeros(a2.shape + (4,), np.uint8)
    nz = a2 > 1e-4
    rgba[..., :3][nz] = np.clip((pre2[nz] / a2[nz][:, None]).round(), 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip((a2 * 255).round(), 0, 255).astype(np.uint8)
    return _png_rgba(rgba)


def _render(elev, theme, mask):
    """elev is rendered at 2× and box-filtered down, which anti-aliases the
    coast. Land is the cleaned coast mask, not the raw 0 m contour, so the
    creeks the coastline layer has filled don't ghost through the wind field
    as unexplained dark lines."""
    p = PALETTES[theme]
    e = _clean(elev)[MARGIN:-MARGIN, MARGIN:-MARGIN]
    land_mask = mask[MARGIN:-MARGIN, MARGIN:-MARGIN]
    depth = np.clip(-e, 0, None)
    t = (0.5 * np.sqrt(np.minimum(depth, SHELF) / SHELF)
         + 0.5 * np.sqrt(np.clip(depth - SHELF, 0, DEEP - SHELF) / (DEEP - SHELF)))
    shallow, deep = np.array(p["shallow"], np.float32), np.array(p["deep"], np.float32)
    sea = shallow + (deep - shallow) * t[..., None]
    rgb = np.where(land_mask[..., None], np.array(p["land"], np.float32), sea)
    h, w, _ = rgb.shape
    rgb = rgb.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
    return _png(np.clip(rgb.round(), 0, 255).astype(np.uint8))


def _variants():
    for th in PALETTES:
        yield th, 1
    for st in COAST:
        for dpr in COAST_HALF_WIDTH:
            yield st, dpr


def _render_variant(elev, style, dpr, z, mask):
    return _render(elev, style, mask) if style in PALETTES else _render_coast(mask, style, dpr, z)


def _path(style, z, x, y, dpr=1):
    name = style if style in PALETTES else f"{style}@{dpr}x"
    return os.path.join(TILE_DIR, name, str(z), str(x), f"{y}.png")


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


_tile_locks: dict = {}
_tile_locks_guard = threading.Lock()


def _tile_lock(z, x, y):
    with _tile_locks_guard:
        return _tile_locks.setdefault((z, x, y), threading.Lock())


def _block_bbox(z, bx, by, size=BLOCK):
    """Mercator bbox of the size×size tile group whose top-left tile is (bx, by)."""
    xmin, _, _, ymax = _tile_bbox(z, bx, by)
    _, ymin, xmax, _ = _tile_bbox(z, bx + size - 1, by + size - 1)
    return xmin, ymin, xmax, ymax


def _render_block(z, bx, by, size=BLOCK):
    """Fetch one size×size group of tiles from NOAA (plus MARGIN around the
    group) and write every style variant of every tile in it. Each tile's
    own margin window is a slice of the group, so the cleanup stays seam-free."""
    px = TILE * 2
    elev = _fetch_elev(_block_bbox(z, bx, by, size), px * size, MARGIN)
    for i in range(size):
        for j in range(size):
            x, y = bx + j, by + i
            if not (0 <= x < 2 ** z and 0 <= y < 2 ** z) or not _in_envelope(z, x, y):
                continue
            sub = elev[i * px:i * px + px + 2 * MARGIN, j * px:j * px + px + 2 * MARGIN]
            mask = _coast_mask(sub, z)               # once per tile, shared by the four coast variants
            for st, d in _variants():
                _write(_path(st, z, x, y, d), _render_variant(sub, st, d, z, mask))


def tile_png(style, z, x, y, dpr=1):
    """PNG bytes for one tile, or None when out of range. One NOAA fetch
    renders every style variant of a BLOCK×BLOCK group of tiles, so a theme
    switch never refetches and a pan over fresh ground costs a quarter of the
    requests; a per-group lock keeps Leaflet's parallel first requests from
    fetching the same group twice. `style` is a PALETTES theme or a COAST
    style (then `dpr` 1 | 2)."""
    dpr = 2 if dpr == 2 else 1
    if style not in STYLES or not (ZMIN <= z <= ZMAX) or not (0 <= x < 2 ** z and 0 <= y < 2 ** z):
        return None
    path = _path(style, z, x, y, dpr)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    if not _in_envelope(z, x, y):
        return None
    bx, by = x - x % BLOCK, y - y % BLOCK
    with _tile_lock(z, bx, by):
        if not os.path.exists(path):
            try:
                _render_block(z, bx, by)
            except Exception as e:
                # the group fetch failed (NOAA hiccup on a 1 MB request): try
                # the one tile that was asked for before giving up
                print(f"[bathy] group {z}/{bx}/{by} failed ({e}); single tile {z}/{x}/{y}", flush=True)
                _render_block(z, x, y, size=1)
    with open(path, "rb") as f:
        return f.read()


def _default_view_tiles():
    """Tiles under the dashboard's default desktop + mobile framing, z 6–9
    (retina fetches one zoom deeper than the view)."""
    lon0, lon1, lat0, lat1 = -75.6, -68.8, 38.6, 43.2
    for z in range(6, 10):
        x0, y0 = _lonlat_to_tile(lon0, lat1, z)
        x1, y1 = _lonlat_to_tile(lon1, lat0, z)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                yield z, x, y


def _region_view_tiles():
    """Tiles under every regional view (desktop and phone framing) at its zoom
    and one deeper for retina — a region's first open shouldn't wait on NOAA."""
    from config import REGION_VIEWS
    for rv in REGION_VIEWS.values():
        for center, zoom in ((rv["center"], rv["zoom"]),
                             (rv.get("mobile_center", rv["center"]), rv.get("mobile_zoom", rv["zoom"]))):
            lat, lon = center
            for z in (zoom, zoom + 1):
                if not (ZMIN <= z <= ZMAX):
                    continue
                half = 520 * 360 / (256 * 2 ** z)          # ≈ half a 1040 px map, in degrees of lon
                lon0, lon1 = lon - half, lon + half
                lat0, lat1 = lat - half * 0.75, lat + half * 0.75
                x0, y0 = _lonlat_to_tile(lon0, lat1, z)
                x1, y1 = _lonlat_to_tile(lon1, lat0, z)
                for x in range(x0, x1 + 1):
                    for y in range(y0, y1 + 1):
                        yield z, x, y


# Every tile of the NY / New England core at the zooms a user pans through
# (6–11; 12+ only under the regional views), rendered once.
CORE = (-76.0, -68.5, 38.3, 43.6)


def _core_tiles():
    lon0, lon1, lat0, lat1 = CORE
    for z in range(6, 12):
        x0, y0 = _lonlat_to_tile(lon0, lat1, z)
        x1, y1 = _lonlat_to_tile(lon1, lat0, z)
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                yield z, x, y


def prewarm(workers=3):
    """The default + regional views first (a fresh install is usable within
    a minute), then the core box; each item is one BLOCK×BLOCK group."""
    done = 0
    for label, tiles in (("views", list(_default_view_tiles()) + list(_region_view_tiles())),
                         ("core", list(_core_tiles()))):
        groups = dict.fromkeys((z, x - x % BLOCK, y - y % BLOCK) for z, x, y in tiles
                               if _in_envelope(z, x, y) and not os.path.exists(_path("dark", z, x, y)))
        if not groups:
            continue
        print(f"[bathy] pre-rendering {label}: {len(groups)} groups of {BLOCK * BLOCK} tiles…", flush=True)
        n = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for ok in ex.map(lambda g: _safe_group(*g), list(groups)):
                n += ok
        print(f"[bathy] pre-render {label} complete: {n}/{len(groups)} groups", flush=True)
        done += n
    return done


def _safe_group(z, bx, by):
    try:
        with _tile_lock(z, bx, by):
            if not os.path.exists(_path("dark", z, bx, by)):
                _render_block(z, bx, by)
        return True
    except Exception as e:
        print(f"[bathy] group {z}/{bx}/{by} failed: {e}", flush=True)
        return False


def prewarm_async():
    threading.Thread(target=prewarm, daemon=True, name="bathy-prewarm").start()
