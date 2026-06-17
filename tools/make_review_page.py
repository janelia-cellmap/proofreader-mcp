#!/usr/bin/env python
"""Generate a self-contained HTML review page for a proofreader collection.

Reads a collection directory (its ``objects.json`` and the verdicts persisted in
``proofreader_state.json``) and writes a single static HTML file with:

* a thumbnail of each object's composite snapshot,
* every field from the verdict (label, confidence, severity, expected_cost,
  flags, recommended_action, rationale, model) plus the bounding-box extents,
* sortable columns (click a header), filters by label / action / model, search,
* per-object Neuroglancer links and an "open all (by category)" link that loads
  one colored mesh layer per label (good=green, overmerged=red, ...).

The page embeds its data inline, so it works opened directly from disk
(``file://``) as long as it sits next to the ``{object_id}.png`` images.

Usage::

    uv run python tools/make_review_page.py ignore/snapshots_lod0_100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from proofreader_mcp.ranking import rank
from proofreader_mcp.storage import Store

# Defaults wired to the c-elegans jrc_P3 layer (override on the CLI).
DEFAULT_MESH_SOURCES = ",".join(
    [
        "https://cellmap-vm1.int.janelia.org/nrs/ackermand/new_meshes/mesh-n-bone/"
        "c-elegans-assign-to-cells-20260603/jrc_P3_E5_D1_N2/mito/"
        "_intermediate_scales.zarr/|zarr2:",
        "https://cellmap-vm1.int.janelia.org/nrs/ackermand/new_meshes/mesh-n-bone/"
        "c-elegans-assign-to-cells-20260603/jrc_P3_E5_D1_N2/mito/multires/"
        "|neuroglancer-precomputed:",
    ]
)
DEFAULT_EM_SOURCE = (
    "https://cellmap-vm1.int.janelia.org/nrs/data/jrc_P3-E5-D1-N2/"
    "jrc_P3-E5-D1-N2.zarr/recon-1/em/fibsem-uint8/|zarr2:"
)
DEFAULT_POSITION = "2108.380126953125,1864.4127197265625,12689.4853515625"

# Per-label mesh colors for the Neuroglancer category layers.
LABEL_COLORS = {
    "acceptable": "#2ca02c",
    "likely_overmerged": "#d62728",
    "likely_oversegmented": "#ff7f0e",
    "misclassified": "#9467bd",
    "junk": "#7f7f7f",
    "uncertain": "#e6c200",
}

PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  body { margin: 0; background: #f6f7f9; color: #1c1f23; }
  header { padding: 14px 20px; background: #fff; border-bottom: 1px solid #e2e5e9;
           position: sticky; top: 0; z-index: 5; }
  h1 { font-size: 18px; margin: 0 0 6px; }
  .summary { font-size: 13px; color: #5a6068; display: flex; flex-wrap: wrap; gap: 12px; }
  .pill { padding: 2px 8px; border-radius: 10px; font-weight: 600; }
  .controls { padding: 10px 20px; display: flex; gap: 12px; align-items: center;
              flex-wrap: wrap; background: #fff; border-bottom: 1px solid #e2e5e9; }
  input, select, button { font-size: 13px; padding: 6px 8px; border: 1px solid #cfd4da;
                  border-radius: 6px; background: #fff; }
  button { cursor: pointer; font-weight: 600; }
  button.ng { background: #1763c7; color: #fff; border-color: #1763c7; }
  table { border-collapse: collapse; width: 100%; background: #fff; }
  th, td { padding: 7px 10px; text-align: left; border-bottom: 1px solid #eceef1;
           font-size: 13px; vertical-align: middle; }
  th { position: sticky; top: 0; background: #f0f2f5; cursor: pointer;
       user-select: none; white-space: nowrap; }
  th.sorted-asc::after { content: " \\2191"; }
  th.sorted-desc::after { content: " \\2193"; }
  td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
  tr:hover td { background: #f9fafb; }
  img.thumb { height: 80px; border: 1px solid #d7dbe0; border-radius: 4px;
              display: block; background: #fff; }
  .rationale { max-width: 420px; color: #3a3f45; }
  .flags { color: #8a5a00; font-size: 12px; }
  a { color: #1763c7; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .lbl-acceptable { background:#e3f4e6; color:#1c7a36; }
  .lbl-likely_overmerged { background:#fde2e1; color:#b3261e; }
  .lbl-likely_oversegmented { background:#fbe7cf; color:#a85a00; }
  .lbl-misclassified { background:#e7e4fb; color:#5b3fb0; }
  .lbl-junk { background:#eceef1; color:#525861; }
  .lbl-uncertain { background:#fff6c2; color:#7a6a00; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="summary" id="summary"></div>
</header>
<div class="controls">
  <input id="search" type="search" placeholder="search id / rationale\\u2026" size="24">
  <select id="labelFilter"><option value="">all labels</option></select>
  <select id="actionFilter"><option value="">all actions</option></select>
  <select id="modelFilter"><option value="">all models</option></select>
  <button class="ng" id="ngAll">Open shown in Neuroglancer (by category)</button>
  <span id="count" style="font-size:13px;color:#5a6068;"></span>
</div>
<table id="tbl">
  <thead><tr>
    <th data-key="thumb" data-type="none">snapshot</th>
    <th data-key="object_id" data-type="str">id</th>
    <th data-key="primary_label" data-type="str">label</th>
    <th class="num" data-key="confidence" data-type="num">conf</th>
    <th class="num" data-key="expected_cost" data-type="num">cost</th>
    <th data-key="recommended_action" data-type="str">action</th>
    <th data-key="flags" data-type="str">flags</th>
    <th data-key="model" data-type="str">model</th>
    <th class="num" data-key="ext_x" data-type="num">x nm</th>
    <th class="num" data-key="ext_y" data-type="num">y nm</th>
    <th class="num" data-key="ext_z" data-type="num">z nm</th>
    <th data-key="rationale" data-type="str">rationale</th>
    <th data-key="ng" data-type="none">ng</th>
  </tr></thead>
  <tbody id="rows"></tbody>
</table>
<script>
__DATA__
let sortKey = "expected_cost", sortDir = -1;

function dims() { const r = NG_RES_NM * 1e-9; return {z:[r,"m"], y:[r,"m"], x:[r,"m"]}; }
function avg(a) { return a.reduce((s,x)=>s+x,0) / a.length; }

// Build a Neuroglancer state: EM image + one segmentation layer per label, each
// colored by category, holding that category's object ids.
function ngState(rows) {
  const layers = [];
  if (NG_EM_SOURCE) layers.push({type:"image", source:NG_EM_SOURCE, name:"em",
    tab:"source", shaderControls:{normalized:{range:[65,154], window:[43,176]}}});
  const byLabel = {};
  for (const d of rows) (byLabel[d.primary_label] = byLabel[d.primary_label] || []).push(d);
  for (const lbl of Object.keys(byLabel)) {
    const ids = byLabel[lbl].map(d => String(d.object_id));
    const colors = {}; ids.forEach(i => colors[i] = LABEL_COLORS[lbl] || "#ffff00");
    layers.push({type:"segmentation", source:NG_MESH_SOURCES, segments:ids,
      segmentColors:colors, name:lbl, tab:"segments"});
  }
  let pos = NG_POSITION;
  const c = rows.filter(d => d.cx != null);
  if (c.length) pos = [avg(c.map(d=>d.cz))/NG_RES_NM, avg(c.map(d=>d.cy))/NG_RES_NM,
                       avg(c.map(d=>d.cx))/NG_RES_NM];
  const state = {dimensions:dims(), position:pos, layers:layers, layout:"4panel",
                 showSlices:false};
  return NG_BASE + "#!" + encodeURIComponent(JSON.stringify(state));
}
function lblPill(l){ return '<span class="pill lbl-'+l+'">'+l+'</span>'; }

function fillSelect(id, vals) {
  const el = document.getElementById(id);
  vals.forEach(v => { const o=document.createElement('option'); o.value=v; o.textContent=v; el.appendChild(o); });
}
function summarize() {
  const counts = {};
  for (const d of DATA) counts[d.primary_label] = (counts[d.primary_label]||0)+1;
  const order = ["likely_overmerged","likely_oversegmented","misclassified","junk","uncertain","acceptable"];
  document.getElementById("summary").innerHTML = '<span>'+DATA.length+' objects</span>' +
    order.filter(l=>counts[l]).map(l=>'<span>'+lblPill(l)+' '+counts[l]+'</span>').join('');
  fillSelect("labelFilter", order.filter(l=>counts[l]));
  fillSelect("actionFilter", [...new Set(DATA.map(d=>d.recommended_action))].sort());
  fillSelect("modelFilter", [...new Set(DATA.map(d=>d.model).filter(Boolean))].sort());
}

function visibleRows() {
  const q = document.getElementById("search").value.toLowerCase();
  const lf = document.getElementById("labelFilter").value;
  const af = document.getElementById("actionFilter").value;
  const mf = document.getElementById("modelFilter").value;
  let rows = DATA.filter(d =>
    (!lf || d.primary_label===lf) && (!af || d.recommended_action===af) &&
    (!mf || d.model===mf) &&
    (!q || (d.object_id+" "+d.rationale+" "+d.flags.join(" ")).toLowerCase().includes(q)));
  rows.sort((a,b)=>{
    let x=a[sortKey], y=b[sortKey];
    if (typeof x==="string"){ x=x.toLowerCase(); y=y.toLowerCase(); }
    return (x<y?-1:x>y?1:0)*sortDir;
  });
  return rows;
}

function render() {
  const rows = visibleRows();
  document.getElementById("count").textContent = rows.length+" shown";
  document.getElementById("rows").innerHTML = rows.map(d => {
    const ng = ngState([d]);
    return '<tr>'+
      '<td><a href="'+d.image+'" target="_blank"><img class="thumb" loading="lazy" src="'+d.image+'"></a></td>'+
      '<td>'+d.object_id+'</td>'+
      '<td>'+lblPill(d.primary_label)+'</td>'+
      '<td class="num">'+d.confidence.toFixed(2)+'</td>'+
      '<td class="num">'+d.expected_cost.toFixed(2)+'</td>'+
      '<td>'+d.recommended_action+'</td>'+
      '<td class="flags">'+d.flags.join(", ")+'</td>'+
      '<td>'+(d.model||"")+'</td>'+
      '<td class="num">'+(d.ext_x??"")+'</td>'+
      '<td class="num">'+(d.ext_y??"")+'</td>'+
      '<td class="num">'+(d.ext_z??"")+'</td>'+
      '<td class="rationale">'+d.rationale+'</td>'+
      '<td><a href="'+ng+'" target="_blank">open</a></td>'+
    '</tr>';
  }).join('');
  document.querySelectorAll("th").forEach(th=>{
    th.className = th.className.replace(/ ?sorted-(asc|desc)/g,"");
    if (th.dataset.key===sortKey) th.className += sortDir<0?" sorted-desc":" sorted-asc";
  });
}

document.querySelectorAll("th").forEach(th=>{
  if (th.dataset.type==="none") return;
  th.addEventListener("click", ()=>{
    if (sortKey===th.dataset.key) sortDir*=-1;
    else { sortKey=th.dataset.key; sortDir = th.dataset.type==="num"?-1:1; }
    render();
  });
});
["search","labelFilter","actionFilter","modelFilter"].forEach(
  id=>document.getElementById(id).addEventListener("input", render));
document.getElementById("ngAll").addEventListener("click",
  ()=>window.open(ngState(visibleRows()), "_blank"));
summarize(); render();
</script>
</body>
</html>
"""


def build_records(collection_dir: Path) -> list[dict]:
    store = Store()
    rep = store.load_collection(collection_dir)
    col = store.get(rep.collection_id)
    ranked = rank(list(col.verdicts.values()))
    records = []
    for it in ranked:
        obj = col.objects[it.object_id]
        meta = obj.meta or {}
        ext = (meta.get("extent_nm") or []) + [None, None, None]
        ctr = (meta.get("centroid_nm") or []) + [None, None, None]
        d = it.to_dict()
        d["image"] = obj.image
        d["model"] = col.verdicts[it.object_id].model_version or ""
        d["ext_x"], d["ext_y"], d["ext_z"] = ext[:3]
        d["cx"], d["cy"], d["cz"] = ctr[:3]
        records.append(d)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "collection_dir", type=Path, help="dir with objects.json + state"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="default: <dir>/review.html"
    )
    parser.add_argument("--title", default=None, help="page title")
    parser.add_argument(
        "--neuroglancer-base",
        default="https://neuroglancer-demo.appspot.com/",
        help="Neuroglancer host to build links against",
    )
    parser.add_argument(
        "--ng-mesh-sources",
        default=DEFAULT_MESH_SOURCES,
        help="comma-separated Neuroglancer source(s) for the mesh/segmentation layers",
    )
    parser.add_argument(
        "--ng-em-source", default=DEFAULT_EM_SOURCE, help="Neuroglancer raw-EM source"
    )
    parser.add_argument(
        "--ng-res-nm", type=float, default=16.0, help="nm per unit of NG position"
    )
    parser.add_argument(
        "--ng-position", default=DEFAULT_POSITION, help="fallback NG position z,y,x"
    )
    args = parser.parse_args(argv)

    records = build_records(args.collection_dir)
    if not records:
        print("no verdicts found in collection; classify objects first")
        return 1

    title = (
        args.title
        or f"proofreader review — {args.collection_dir.name} ({len(records)})"
    )
    mesh_sources = [s for s in args.ng_mesh_sources.split(",") if s]
    position = [float(x) for x in args.ng_position.split(",")]
    data_js = (
        "const DATA = " + json.dumps(records) + ";\n"
        "const NG_BASE = " + json.dumps(args.neuroglancer_base) + ";\n"
        "const NG_MESH_SOURCES = " + json.dumps(mesh_sources) + ";\n"
        "const NG_EM_SOURCE = " + json.dumps(args.ng_em_source) + ";\n"
        "const NG_RES_NM = " + json.dumps(args.ng_res_nm) + ";\n"
        "const NG_POSITION = " + json.dumps(position) + ";\n"
        "const LABEL_COLORS = " + json.dumps(LABEL_COLORS) + ";"
    )
    html = PAGE_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data_js)

    out = args.output or args.collection_dir / "review.html"
    out.write_text(html)
    print(f"wrote {out}  ({len(records)} objects)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
