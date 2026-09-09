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
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from cache import record_api_calls, _DISK_CACHE_DIR

STYLE = "v1"
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
    "dark":  {"land": (0x23, 0x23, 0x30), "shallow": (0x12, 0x12, 0x19), "deep": (0x05, 0x05, 0x0a)},
    "light": {"land": (0xf7, 0xf7, 0xf9), "shallow": (0xdc, 0xe1, 0xea), "deep": (0xb0, 0xbb, 0xcd)},
}

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


def _fetch_elev(bbox, px):
    params = {
        "bbox": ",".join(f"{v:.3f}" for v in bbox), "bboxSR": 3857, "imageSR": 3857,
        "size": f"{px},{px}", "format": "tiff", "pixelType": "F32",
        "interpolation": "RSP_BilinearInterpolation", "f": "image",
    }
    for attempt in (1, 2):
        try:
            r = requests.get(NOAA_URL, params=params, timeout=60)
            r.raise_for_status()
            record_api_calls("bathy_tiles", 1)
            elev = _parse_tiff_f32(r.content)
            break
        except Exception:
            if attempt == 2:
                raise
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


def _render(elev, theme):
    """elev is rendered at 2× and box-filtered down, which anti-aliases the coast."""
    p = PALETTES[theme]
    e = np.nan_to_num(elev.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    e[e < -20000] = 0.0                  # nodata sentinel
    depth = np.clip(-e, 0, None)
    t = (0.5 * np.sqrt(np.minimum(depth, SHELF) / SHELF)
         + 0.5 * np.sqrt(np.clip(depth - SHELF, 0, DEEP - SHELF) / (DEEP - SHELF)))
    shallow, deep = np.array(p["shallow"], np.float32), np.array(p["deep"], np.float32)
    sea = shallow + (deep - shallow) * t[..., None]
    rgb = np.where((e >= 0)[..., None], np.array(p["land"], np.float32), sea)
    h, w, _ = rgb.shape
    rgb = rgb.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
    return _png(np.clip(rgb.round(), 0, 255).astype(np.uint8))


def _path(theme, z, x, y):
    return os.path.join(TILE_DIR, theme, str(z), str(x), f"{y}.png")


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


def tile_png(theme, z, x, y):
    """PNG bytes for one tile, or None when out of range. One NOAA fetch
    renders both themes, so a theme switch never refetches; a per-tile lock
    keeps Leaflet's parallel first requests from fetching the same tile twice."""
    if theme not in PALETTES or not (ZMIN <= z <= ZMAX) or not (0 <= x < 2 ** z and 0 <= y < 2 ** z):
        return None
    path = _path(theme, z, x, y)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    if not _in_envelope(z, x, y):
        return None
    with _tile_lock(z, x, y):
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        elev = _fetch_elev(_tile_bbox(z, x, y), TILE * 2)
        out = None
        for th in PALETTES:
            png = _render(elev, th)
            _write(_path(th, z, x, y), png)
            if th == theme:
                out = png
    return out


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


def prewarm(workers=4):
    todo = [(z, x, y) for z, x, y in _default_view_tiles()
            if not os.path.exists(_path("dark", z, x, y))]
    if not todo:
        return 0
    print(f"[bathy] pre-rendering {len(todo)} tiles…", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for ok in ex.map(lambda t: _safe_tile(*t), todo):
            done += ok
    print(f"[bathy] pre-render complete: {done}/{len(todo)}", flush=True)
    return done


def _safe_tile(z, x, y):
    try:
        return tile_png("dark", z, x, y) is not None
    except Exception as e:
        print(f"[bathy] tile {z}/{x}/{y} failed: {e}", flush=True)
        return False


def prewarm_async():
    threading.Thread(target=prewarm, daemon=True, name="bathy-prewarm").start()
