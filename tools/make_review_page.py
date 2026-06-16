#!/usr/bin/env python
"""Generate a self-contained HTML review page for a proofreader collection.

Reads a collection directory (its ``objects.json`` and the verdicts persisted in
``proofreader_state.json``) and writes a single static HTML file with:

* a thumbnail of each object's composite snapshot,
* every field from the verdict (label, confidence, severity, expected_cost,
  flags, recommended_action, rationale) plus the bounding-box extents,
* sortable columns (click a header) and a label filter + text search,
* an optional per-object Neuroglancer link.

The page embeds its data inline, so it works opened directly from disk
(``file://``) as long as it sits next to the ``{object_id}.png`` images — by
default it is written into the collection directory itself.

Usage::

    uv run python tools/make_review_page.py ignore/snapshots_lod0_100 \
        --neuroglancer-source precomputed://file:///nrs/.../mito/multires
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from proofreader_mcp.ranking import rank
from proofreader_mcp.storage import Store

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
  input, select { font-size: 13px; padding: 6px 8px; border: 1px solid #cfd4da;
                  border-radius: 6px; }
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
  .rationale { max-width: 460px; color: #3a3f45; }
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
  <input id="search" type="search" placeholder="search id / rationale\\u2026" size="28">
  <select id="labelFilter"><option value="">all labels</option></select>
  <select id="actionFilter"><option value="">all actions</option></select>
  <span id="count" style="font-size:13px;color:#5a6068;"></span>
</div>
<table id="tbl">
  <thead><tr>
    <th data-key="thumb" data-type="none">snapshot</th>
    <th data-key="object_id" data-type="str">id</th>
    <th data-key="primary_label" data-type="str">label</th>
    <th class="num" data-key="confidence" data-type="num">conf</th>
    <th class="num" data-key="severity" data-type="num">sev</th>
    <th class="num" data-key="expected_cost" data-type="num">cost</th>
    <th data-key="recommended_action" data-type="str">action</th>
    <th data-key="flags" data-type="str">flags</th>
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

function ngLink(d) {
  const layers = [];
  if (RAW_SOURCE) layers.push({type:"image", source:RAW_SOURCE, name:"em"});
  const seg = SEG_SOURCE || NG_SOURCE;
  if (seg) layers.push({type:"segmentation", source:seg, segments:[String(d.object_id)], name:NG_NAME});
  if (!layers.length) return "";
  const state = {
    dimensions:{x:[1e-9,"m"],y:[1e-9,"m"],z:[1e-9,"m"]},
    layers, layout:"4panel",
    selectedLayer:{visible:true, layer:NG_NAME}
  };
  if (d.cx != null) state.position = [d.cx, d.cy, d.cz];
  return NG_BASE + "#!" + encodeURIComponent(JSON.stringify(state));
}
function lblPill(l){ return '<span class="pill lbl-'+l+'">'+l+'</span>'; }

function summarize() {
  const counts = {};
  for (const d of DATA) counts[d.primary_label] = (counts[d.primary_label]||0)+1;
  const order = ["likely_overmerged","likely_oversegmented","misclassified","junk","uncertain","acceptable"];
  const el = document.getElementById("summary");
  el.innerHTML = '<span>'+DATA.length+' objects</span>' +
    order.filter(l=>counts[l]).map(l=>'<span>'+lblPill(l)+' '+counts[l]+'</span>').join('');
  const lf = document.getElementById("labelFilter");
  order.filter(l=>counts[l]).forEach(l=>{ const o=document.createElement('option'); o.value=l; o.textContent=l+' ('+counts[l]+')'; lf.appendChild(o); });
  const acts = [...new Set(DATA.map(d=>d.recommended_action))].sort();
  const af = document.getElementById("actionFilter");
  acts.forEach(a=>{ const o=document.createElement('option'); o.value=a; o.textContent=a; af.appendChild(o); });
}

function visibleRows() {
  const q = document.getElementById("search").value.toLowerCase();
  const lf = document.getElementById("labelFilter").value;
  const af = document.getElementById("actionFilter").value;
  let rows = DATA.filter(d =>
    (!lf || d.primary_label===lf) && (!af || d.recommended_action===af) &&
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
  const body = rows.map(d => {
    const ng = ngLink(d);
    return '<tr>'+
      '<td><a href="'+d.image+'" target="_blank"><img class="thumb" loading="lazy" src="'+d.image+'"></a></td>'+
      '<td>'+d.object_id+'</td>'+
      '<td>'+lblPill(d.primary_label)+'</td>'+
      '<td class="num">'+d.confidence.toFixed(2)+'</td>'+
      '<td class="num">'+d.severity.toFixed(2)+'</td>'+
      '<td class="num">'+d.expected_cost.toFixed(2)+'</td>'+
      '<td>'+d.recommended_action+'</td>'+
      '<td class="flags">'+d.flags.join(", ")+'</td>'+
      '<td class="num">'+(d.ext_x??"")+'</td>'+
      '<td class="num">'+(d.ext_y??"")+'</td>'+
      '<td class="num">'+(d.ext_z??"")+'</td>'+
      '<td class="rationale">'+d.rationale+'</td>'+
      '<td>'+(ng?'<a href="'+ng+'" target="_blank">open</a>':'')+'</td>'+
    '</tr>';
  }).join('');
  document.getElementById("rows").innerHTML = body;
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
["search","labelFilter","actionFilter"].forEach(id=>document.getElementById(id).addEventListener("input", render));
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
        "--raw-source",
        default="",
        help="Neuroglancer image-layer source for the raw EM, e.g. "
        "'https://host/.../em/fibsem-uint8/|zarr2:'",
    )
    parser.add_argument(
        "--seg-source",
        default="",
        help="Neuroglancer segmentation-layer source (voxel seg), e.g. "
        "'https://host/.../mito/|zarr2:'. Each row's link selects that object's id.",
    )
    parser.add_argument(
        "--neuroglancer-source",
        default="",
        help="fallback segmentation source if --seg-source is not given (e.g. the "
        "precomputed mesh layer)",
    )
    parser.add_argument(
        "--neuroglancer-base",
        default="https://neuroglancer-demo.appspot.com/",
        help="Neuroglancer host to build links against",
    )
    parser.add_argument("--neuroglancer-name", default="mito", help="NG seg layer name")
    args = parser.parse_args(argv)

    records = build_records(args.collection_dir)
    if not records:
        print("no verdicts found in collection; classify objects first")
        return 1

    title = (
        args.title
        or f"proofreader review — {args.collection_dir.name} ({len(records)})"
    )
    data_js = (
        "const DATA = " + json.dumps(records) + ";\n"
        "const NG_BASE = " + json.dumps(args.neuroglancer_base) + ";\n"
        "const RAW_SOURCE = " + json.dumps(args.raw_source) + ";\n"
        "const SEG_SOURCE = " + json.dumps(args.seg_source) + ";\n"
        "const NG_SOURCE = " + json.dumps(args.neuroglancer_source) + ";\n"
        "const NG_NAME = " + json.dumps(args.neuroglancer_name) + ";"
    )
    html = PAGE_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data_js)

    out = args.output or args.collection_dir / "review.html"
    out.write_text(html)
    print(f"wrote {out}  ({len(records)} objects)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
