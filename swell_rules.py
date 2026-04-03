"""
colesurfs — Swell Categorization Rules
═══════════════════════════════════════════
Source of truth for category thresholds: swell-categorization-scheme.toml
Source of truth for category colors:     COLORS dict below

To update thresholds: edit the .toml file, then Refresh in the app (or restart).
To update colors:     edit COLORS below and restart.
"""
import os
try:
    import tomllib                        # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib           # pip install tomli
    except ImportError:
        tomllib = None

# ── Category order: low → high ──────────────────────────────────────────────
CATEGORIES = ['FLAT', 'WEAK', 'FUN', 'SOLID', 'FIRING', 'HECTIC', 'MONSTRO']

# ── Colors per category ──────────────────────────────────────────────────────
# dark_bg / dark_text  →  dark mode cell background and text
# light_bg / light_text →  light mode cell background and text
COLORS = {
    'FLAT':    dict(dark_bg='#131316', dark_text='#404055',
                    light_bg='#e2e2de', light_text='#70707c'),
    'WEAK':    dict(dark_bg='#0d1520', dark_text='#3d7ab5',
                    light_bg='#d8e8f8', light_text='#1a5a9a'),
    'FUN':     dict(dark_bg='#0d1a0f', dark_text='#3fb950',
                    light_bg='#ccecd4', light_text='#166028'),
    'SOLID':   dict(dark_bg='#1a1800', dark_text='#f5c518',   # bright gold — clearly distinct from FIRING red
                    light_bg='#f5e6c0', light_text='#7a5500'),
    'FIRING':  dict(dark_bg='#220a0a', dark_text='#ff5a4e',  # deeper red bg, vivid red text
                    light_bg='#f8d8d8', light_text='#a81010'),
    'HECTIC':  dict(dark_bg='#1c1240', dark_text='#7c6af7',   # current GOOD blue/violet
                    light_bg='#e2dcf8', light_text='#3828b8'),
    'MONSTRO': dict(dark_bg='#1a0a30', dark_text='#c084fc',   # deeper purple
                    light_bg='#ecdcf8', light_text='#7010b8'),
}

_BANDS_CACHE = None   # populated by load_bands(); cleared by reload()

_TOML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'swell-categorization-scheme.toml'
)


def _parse_value(val):
    """
    Normalise one TOML category value into the internal rule format.
    TOML already converts numbers to float; strings arrive as-is.

    Returns:
      'always'        — period alone determines this category
      'never'         — category not applicable for this period band
      float           — upper bound (height_ft < N → this category)
      {'gte': float}  — lower bound (height_ft >= N → this category)
    """
    if isinstance(val, (int, float)):
        return float(val)
    v = str(val).strip().lower()
    if v == 'always':
        return 'always'
    if v == 'never':
        return 'never'
    if v.startswith('>='):
        return {'gte': float(v[2:])}
    try:
        return float(v)
    except ValueError:
        return 'never'   # graceful fallback


def load_bands(force: bool = False):
    """
    Parse swell-categorization-scheme.toml and return a list of period bands,
    sorted ascending by period_ub:

      [{'period_ub': float | None,   ← None means ≥ last explicit bound (infinity)
        'rules': {CAT: value, ...}}, ...]

    Cached after first load. Pass force=True (or call reload()) to re-read from disk.
    """
    global _BANDS_CACHE
    if _BANDS_CACHE is not None and not force:
        return _BANDS_CACHE

    if tomllib is None:
        raise RuntimeError(
            "No TOML library available. Run: pip install tomli   "
            "(or upgrade to Python 3.11+)"
        )

    with open(_TOML_PATH, 'rb') as f:
        data = tomllib.load(f)

    bands = []
    for entry in data.get('band', []):
        raw_ub = entry['period_upper_bound']

        # "inf" string or the actual float infinity → catch-all band
        if isinstance(raw_ub, str) and raw_ub.lower() == 'inf':
            period_ub   = None
            sort_key    = float('inf')
        else:
            period_ub   = float(raw_ub)
            sort_key    = period_ub

        rules = {cat: _parse_value(entry.get(cat, 'never')) for cat in CATEGORIES}

        bands.append({
            'period_ub': period_ub,
            '_sort_key': sort_key,
            'rules':     rules,
        })

    bands.sort(key=lambda b: b['_sort_key'])
    for b in bands:
        del b['_sort_key']

    _BANDS_CACHE = bands
    return bands


def reload():
    """Force re-read from disk. Called by /api/refresh so edits to the .toml take effect."""
    return load_bands(force=True)


def categorize(height_ft, period_s):
    """
    Return the category name (e.g. 'FUN') for a swell component.

    Algorithm:
      1. Find the first period band whose period_ub > period_s (or the last band).
      2. Walk CATEGORIES in order (low → high).
         - 'never'  → skip
         - 'always' → return this category immediately
         - {'gte':N}→ return this category if height_ft >= N
         - float N  → return this category if height_ft < N (upper bound)
      3. If no explicit match, return the last non-'never' category seen
         (catches heights that exceed all upper bounds in the row).
    """
    if height_ft is None or period_s is None:
        return 'FLAT'

    bands = load_bands()
    band = bands[-1]
    for b in bands:
        ub = b['period_ub']
        if ub is None:          # catch-all / infinity band
            band = b
            break
        if period_s < ub:
            band = b
            break

    rules      = band['rules']
    last_valid = CATEGORIES[0]   # safe fallback

    for cat in CATEGORIES:
        rule = rules.get(cat, 'never')
        if rule == 'never':
            continue
        last_valid = cat
        if rule == 'always':
            break
        if isinstance(rule, dict):           # {'gte': N}
            if height_ft >= rule['gte']:
                break
        elif isinstance(rule, float):        # upper bound N
            if height_ft < rule:
                break
        # else: height exceeds this upper bound — keep walking

    return last_valid
