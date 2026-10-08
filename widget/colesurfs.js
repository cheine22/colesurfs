// colesurfs — Fun+ Days home-screen widget (Scriptable)
//
// Served at https://surfreport.coleheine.com/widget/colesurfs.js. The script
// installed on the phone is a two-line loader, so edits here ship through
// git + autopull without touching the phone:
//
//   const r = new Request("https://surfreport.coleheine.com/widget/colesurfs.js");
//   await eval(await r.loadString());
//
// `eval` parses a classic script, where top-level await is a syntax error, so
// everything below runs inside one async function and the loader awaits it.
//
// Widget parameter (long-press → Edit Widget):
//   (empty) or "forecast-nyc"  Fun+ Days, regions in regions.yaml order
//   "forecast-bi"              Fun+ Days, Block Island Sound first, then the rest
//                              (small shows 1 region, medium 2, large 4)
//   "live-lido"                live widget: NY Harbor Entrance buoy + Lido Beach
//   "live-landing"             live widget: Block Island Sound buoy + The Landing
//   "live-southampton"         live widget: Block Island Sound buoy + Road G
//                              (live widgets are medium only)
//   "calibrate"                a point ruler to read the phone's widget size
// Append "; size=169x169" to any of these to pin the widget's point size when
// the built-in table guesses wrong for a phone.
//
// The widget shows a PNG the server renders from the mockup's own HTML/CSS
// (/widget/tile.png → templates/widget_render.html via headless Chrome), so
// what appears on the phone is the artifact, pixel for pixel. Scriptable only
// fetches the image, sizes it, and handles the tap.
//
// Sharpness: inside a widget Scriptable recompresses any image it LOADS above
// roughly 500 k px (talk.automators.fm/t/low-quality-png-in-widget/10334 —
// a 507×507 small survives, a 1080×507 medium does not), so the phone never
// loads the full PNG: the server renders it in tiles under that limit and the
// widget fetches each tile and lays them edge to edge at 1:1.
//
// Tap target: the server says where each widget opens (X-Tap-Url on every
// tile): the Fun+ Days widget → the dashboard, live-lido → Lido Beach's
// Surfline page, the same link as the spot name in the regional view. iOS
// gives a home-screen web app no URL scheme and Shortcuts' Open App won't
// target one either (tried 2026-10), so these open in Safari; set
// OPEN_SHORTCUT to a Shortcut name to run that instead.

(async () => {
const SITE = "https://surfreport.coleheine.com";
const OPEN_SHORTCUT = "";                        // "" → open the site in Safari
const DEFAULT_REGIONS = ["NY Harbor Entrance", "Block Island Sound", "Massachusetts", "Barnegat"];
const PREVIEW_FAMILY = "medium";               // when run inside the Scriptable app
const REFRESH_MIN = 30;
const MAX_TILE_PX = 290;

const FAMILY = config.widgetFamily || PREVIEW_FAMILY;
const N_BY_FAMILY = { small: 1, medium: 2, large: 4 };
const APPEARANCE = Device.isUsingDarkAppearance() ? "dark" : "light";

// ── parameter ───────────────────────────────────────────────────────────────
const rawParam = (args.widgetParameter || "").trim();
const CALIBRATE = rawParam.toLowerCase() === "calibrate";
let sizeOverride = null;
const param = rawParam.split(";").map(s => s.trim()).filter(p => {
  const m = p.match(/^size\s*=\s*(\d+)\s*x\s*(\d+)$/i);
  if (m) { sizeOverride = { w: +m[1], h: +m[2] }; return false; }
  return true;
}).join(";");
const LIVE_SPOTS = { "live-lido": "Lido Beach", "live-landing": "The Landing", "live-southampton": "Road G" };
const P = param.toLowerCase();
const KIND = P in LIVE_SPOTS ? "live" : "forecast";
const LIVE_SPOT = LIVE_SPOTS[P] || null;
let regions = P === "forecast-bi"
  ? ["Block Island Sound", ...DEFAULT_REGIONS.filter(r => r !== "Block Island Sound")]
  : DEFAULT_REGIONS;                           // empty, forecast-nyc, anything unknown
regions = regions.slice(0, N_BY_FAMILY[FAMILY] || 4);

// ── geometry ────────────────────────────────────────────────────────────────
// WidgetKit point sizes by screen width (no API exposes them to the script).
// 402-wide Pro phones (16/17/18 Pro) get the 169-pt class; 158 left bars.
function widgetSize() {
  const sw = Device.screenSize().width;
  const table = [[320, 141, 292, 311], [375, 155, 329, 345], [390, 158, 338, 354],
                 [393, 158, 338, 354], [402, 169, 360, 379], [414, 169, 360, 379],
                 [428, 170, 364, 382], [430, 170, 364, 382], [440, 170, 364, 382]];
  let best = table[2];
  for (const t of table) if (Math.abs(t[0] - sw) < Math.abs(best[0] - sw)) best = t;
  const [, s, mw, lh] = best;
  if (FAMILY === "small") return { w: s, h: s };
  if (FAMILY === "large") return { w: mw, h: lh };
  return { w: mw, h: s };
}
const SIZE = sizeOverride || widgetSize();
const SCALE = Math.round(Device.screenScale()) || 3;

function splitPts(total, n) {                  // integer point widths that sum exactly (server mirrors this)
  const base = Math.floor(total / n), out = [];
  for (let i = 0; i < n; i++) out.push(base + (i < total - base * n ? 1 : 0));
  return out;
}
const COLS = Math.ceil(SIZE.w * SCALE / MAX_TILE_PX), ROWS = Math.ceil(SIZE.h * SCALE / MAX_TILE_PX);
const XS = splitPts(SIZE.w, COLS), YS = splitPts(SIZE.h, ROWS);

// ── tiles ───────────────────────────────────────────────────────────────────
const fm = FileManager.local();
const slug = KIND === "live" ? "live-" + LIVE_SPOT.replace(/[^A-Za-z]+/g, "_") : regions.join("+").replace(/[^A-Za-z]+/g, "_");

function tileUrl(col, row) {
  return `${SITE}/widget/tile.png?family=${FAMILY}&appearance=${APPEARANCE}&kind=${KIND}`
       + (KIND === "live" ? `&spot=${encodeURIComponent(LIVE_SPOT)}` : `&regions=${encodeURIComponent(regions.join(","))}`)
       + `&w=${SIZE.w}&h=${SIZE.h}&scale=${SCALE}&cols=${COLS}&rows=${ROWS}&col=${col}&row=${row}`;
}
function tapPath() {
  return fm.joinPath(fm.documentsDirectory(), `colesurfs-${FAMILY}-${slug}-tap.txt`);
}
function tilePath(col, row) {
  return fm.joinPath(fm.documentsDirectory(), `colesurfs-${FAMILY}-${slug}-${APPEARANCE}-${COLS}x${ROWS}-${col}-${row}.png`);
}

// All tiles in parallel; on any failure fall back to the last cached set.
let tapUrl = null;                             // from the tiles' X-Tap-Url header
async function fetchTiles() {
  const jobs = [];
  for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) jobs.push((async () => {
    const req = new Request(tileUrl(c, r)); req.timeoutInterval = 40;
    const img = await req.loadImage();
    if (!img || !img.size.width) throw new Error("no image");
    const hdr = (req.response && req.response.headers) || {};
    const u = hdr["X-Tap-Url"] || hdr["x-tap-url"];
    if (u) { tapUrl = u; try { fm.writeString(tapPath(), u); } catch (e) {} }
    return { img, c, r };
  })());
  try {
    const got = await Promise.all(jobs);
    for (const t of got) fm.writeImage(tilePath(t.c, t.r), t.img);
    return { tiles: got, stale: false };
  } catch (e) {
    const cached = [];
    for (let r = 0; r < ROWS; r++) for (let c = 0; c < COLS; c++) {
      const pth = tilePath(c, r);
      if (!fm.fileExists(pth)) return { tiles: null, stale: true };
      cached.push({ img: fm.readImage(pth), c, r });
    }
    return { tiles: cached, stale: true };
  }
}

// ── widgets ─────────────────────────────────────────────────────────────────
const openUrl = OPEN_SHORTCUT
  ? `shortcuts://run-shortcut?name=${encodeURIComponent(OPEN_SHORTCUT)}`
  : SITE + "/";

// A fetched tile is a scale-1 image that SwiftUI would shrink 3× into its
// frame, and its edge sampling fades each border a pixel — a dark hairline
// between tiles. Redrawn into a screen-scale context it becomes a true 3×
// image drawn 1:1, so nothing is resampled and the tiles meet seamlessly.
function native(img, wPt, hPt) {
  const ctx = new DrawContext();
  ctx.size = new Size(wPt, hPt); ctx.opaque = true; ctx.respectScreenScale = true;
  ctx.drawImageInRect(img, new Rect(0, 0, wPt, hPt));
  return ctx.getImage();
}

function buildWidget(tiles, stale) {
  const w = new ListWidget();
  let url = openUrl;
  if (!OPEN_SHORTCUT) {
    if (!tapUrl) { try { if (fm.fileExists(tapPath())) tapUrl = fm.readString(tapPath()); } catch (e) {} }
    if (tapUrl) url = tapUrl;
  }
  w.url = url;
  w.refreshAfterDate = new Date(Date.now() + (stale ? 10 : REFRESH_MIN) * 60 * 1000);
  w.setPadding(0, 0, 0, 0); w.spacing = 0;
  w.backgroundColor = new Color(APPEARANCE === "dark" ? "#131316" : "#ffffff");
  const byPos = {};
  for (const t of tiles) byPos[`${t.c},${t.r}`] = t.img;
  const v = w.addStack(); v.layoutVertically(); v.spacing = 0; v.setPadding(0, 0, 0, 0);
  for (let r = 0; r < ROWS; r++) {
    const h = v.addStack(); h.layoutHorizontally(); h.spacing = 0; h.setPadding(0, 0, 0, 0);
    for (let c = 0; c < COLS; c++) {
      const wi = h.addImage(native(byPos[`${c},${r}`], XS[c], YS[r]));
      wi.imageSize = new Size(XS[c], YS[r]); wi.resizable = false;
    }
  }
  return w;
}

// "calibrate": a point ruler from the top-left corner. The last label visible
// at the right edge is the widget's width in points, at the bottom its height.
function calibrateWidget() {
  const N = 400;
  const ctx = new DrawContext();
  ctx.size = new Size(N, N); ctx.opaque = true; ctx.respectScreenScale = false;
  ctx.setFillColor(new Color("#131316")); ctx.fillRect(new Rect(0, 0, N, N));
  ctx.setFont(Font.boldMonospacedSystemFont(7)); ctx.setTextColor(new Color("#3fb950"));
  for (let p = 0; p <= N; p += 5) {
    const major = p % 10 === 0, len = major ? 8 : 4;
    ctx.setFillColor(new Color(major ? "#e8e8f0" : "#606075"));
    ctx.fillRect(new Rect(p, 0, 1, len)); ctx.fillRect(new Rect(0, p, len, 1));
    if (major && p % 20 === 0 && p > 0) {
      ctx.drawText(String(p), new Point(p - 6, 9));
      ctx.drawText(String(p), new Point(9, p - 4));
    }
  }
  ctx.setTextColor(new Color("#a0a0b8"));
  ctx.drawText("read the last visible number at the right and bottom edges", new Point(24, 60));
  ctx.drawText(`table guess: ${widgetSize().w}x${widgetSize().h}  scale ${SCALE}`, new Point(24, 72));
  const w = new ListWidget();
  w.setPadding(0, 0, 0, 0); w.backgroundColor = new Color("#131316");
  const v = w.addStack(); v.layoutVertically(); v.setPadding(0, 0, 0, 0);
  const h = v.addStack(); h.layoutHorizontally(); h.setPadding(0, 0, 0, 0);
  const wi = h.addImage(ctx.getImage()); wi.imageSize = new Size(N, N); wi.resizable = true;
  h.addSpacer(); v.addSpacer();
  return w;
}

function errorWidget(msg) {
  const w = new ListWidget();
  w.url = openUrl;
  w.backgroundColor = new Color(APPEARANCE === "dark" ? "#131316" : "#ffffff");
  const ink = new Color(APPEARANCE === "dark" ? "#e8e8f0" : "#1e1e21");
  const t = w.addText("colesurfs"); t.font = Font.boldMonospacedSystemFont(12); t.textColor = ink;
  const m = w.addText(msg); m.font = Font.mediumMonospacedSystemFont(9); m.textColor = ink; m.textOpacity = 0.6;
  w.refreshAfterDate = new Date(Date.now() + 10 * 60 * 1000);
  return w;
}

// ── run ─────────────────────────────────────────────────────────────────────
let widget;
if (CALIBRATE) {
  widget = calibrateWidget();
} else if (KIND === "live" && FAMILY !== "medium") {
  widget = errorWidget("the live widget is medium only");
} else {
  const { tiles, stale } = await fetchTiles();
  widget = tiles ? buildWidget(tiles, stale) : errorWidget("no data yet — check the connection");
}
if (config.runsInWidget) {
  Script.setWidget(widget);
} else {
  if (FAMILY === "small") await widget.presentSmall();
  else if (FAMILY === "large") await widget.presentLarge();
  else await widget.presentMedium();
}
Script.complete();
})();
