from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "result2.json"
CONTEXT_PATH = ROOT / "test2.json"
OUTPUT_PATH = ROOT / "result_visual2.html"


def main() -> None:
    payload = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if CONTEXT_PATH.exists():
        payload["_previewContext"] = json.loads(
            CONTEXT_PATH.read_text(encoding="utf-8")
        )
    data_json = json.dumps(payload, ensure_ascii=False)
    OUTPUT_PATH.write_text(_html(data_json), encoding="utf-8")
    print(OUTPUT_PATH)


def _html(data_json: str) -> str:
    escaped = html.escape(data_json, quote=False)
    return f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Layout Preview</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f4ef; color: #242424; }}
    header {{ display: flex; gap: 12px; align-items: center; padding: 12px 16px; border-bottom: 1px solid #ddd7ca; }}
    select {{ font-size: 14px; padding: 6px 8px; }}
    main {{ display: grid; grid-template-columns: 1fr 320px; min-height: calc(100vh - 57px); }}
    canvas {{ width: 100%; height: calc(100vh - 57px); background: #fffdf7; }}
    aside {{ border-left: 1px solid #ddd7ca; padding: 14px; overflow: auto; background: #fbfaf7; }}
    .row {{ display: grid; grid-template-columns: 18px 1fr; gap: 8px; align-items: center; margin: 8px 0; font-size: 13px; }}
    .swatch {{ width: 14px; height: 14px; border: 1px solid #999; }}
    .muted {{ color: #666; }}
  </style>
</head>
<body>
  <script id="payload" type="application/json">{escaped}</script>
  <header>
    <strong>Layout Preview</strong>
    <select id="option"></select>
    <span id="meta" class="muted"></span>
  </header>
  <main>
    <canvas id="canvas"></canvas>
    <aside>
      <strong>Objects</strong>
      <div id="legend"></div>
    </aside>
  </main>
  <script>
    const payload = JSON.parse(document.getElementById("payload").textContent);
    const previewContext = payload._previewContext || {{}};
    const room = payload.room || previewContext.room || {{}};
    const walls = payload.walls || previewContext.walls || [];
    const options = payload.options?.length
      ? payload.options
      : [{{ optionId: payload.selectedOptionId || "variant_1", label: "Option 1", objects: payload.objects || [], openings: payload.openings || [] }}];
    const select = document.getElementById("option");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const legend = document.getElementById("legend");
    const meta = document.getElementById("meta");

    options.forEach((option, index) => {{
      const item = document.createElement("option");
      item.value = String(index);
      item.textContent = `${{option.label || option.optionId}} (${{option.objects?.length || 0}} objects)`;
      select.appendChild(item);
    }});
    select.addEventListener("change", draw);
    window.addEventListener("resize", draw);
    draw();

    function draw() {{
      const option = options[Number(select.value || 0)];
      const objects = option.objects || [];
      const openings = option.openings || [];
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(canvas.clientWidth * dpr);
      canvas.height = Math.floor(canvas.clientHeight * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);

      const bounds = computeBounds(objects, openings, walls, room);
      const pad = 50;
      const sx = (canvas.clientWidth - pad * 2) / Math.max(1, bounds.maxX - bounds.minX);
      const sz = (canvas.clientHeight - pad * 2) / Math.max(1, bounds.maxZ - bounds.minZ);
      const scale = Math.min(sx, sz);
      const toScreen = (x, z) => ({{
        x: pad + (x - bounds.minX) * scale,
        y: canvas.clientHeight - pad - (z - bounds.minZ) * scale,
      }});

      drawGrid(bounds, scale, toScreen);
      drawRoom(room, toScreen);
      drawWalls(walls, toScreen, scale);
      openings.forEach((item) => drawOpening(item, toScreen, scale));
      objects.forEach((item, index) => drawObject(item, index, toScreen, scale));
      renderLegend(objects);
      meta.textContent = `score=${{option.layoutScore ?? "n/a"}} hardValid=${{option.hardValid ?? "n/a"}}`;
    }}

    function computeBounds(objects, openings, walls, room) {{
      const xs = [0], zs = [0];
      const polygons = room.polygons || room.polygon || room.polygon_ccw || [];
      for (const point of polygons) {{
        if (Array.isArray(point)) {{
          xs.push(Number(point[0] || 0));
          zs.push(Number(point[1] || 0));
        }} else if (point && typeof point === "object") {{
          xs.push(Number(point.x || 0));
          zs.push(Number(point.y ?? point.z ?? 0));
        }}
      }}
      for (const wall of walls) {{
        const a = wall.startPoint || wall.start || wall.a;
        const b = wall.endPoint || wall.end || wall.b;
        for (const point of [a, b]) {{
          if (!Array.isArray(point)) continue;
          xs.push(Number(point[0] || 0));
          zs.push(Number(point[1] || 0));
        }}
      }}
      for (const item of objects) {{
        const p = item.position || {{}};
        const s = item.size || [0.5, 0.5, 0.5];
        xs.push(Number(p.x || 0) - s[0] / 2, Number(p.x || 0) + s[0] / 2);
        zs.push(Number(p.z || 0) - s[2] / 2, Number(p.z || 0) + s[2] / 2);
      }}
      for (const item of openings) {{
        const p = item.position || [0, 0, 0];
        xs.push(Number(p[0] || 0));
        zs.push(Number(p[2] || 0));
      }}
      return {{
        minX: Math.min(...xs) - 0.5,
        maxX: Math.max(...xs) + 0.5,
        minZ: Math.min(...zs) - 0.5,
        maxZ: Math.max(...zs) + 0.5,
      }};
    }}

    function drawRoom(room, toScreen) {{
      const polygons = room.polygons || room.polygon || room.polygon_ccw || [];
      const points = polygons.map(point2).filter(Boolean);
      if (points.length < 3) return;
      ctx.save();
      ctx.beginPath();
      points.forEach((p, index) => {{
        const s = toScreen(p.x, p.z);
        if (index === 0) ctx.moveTo(s.x, s.y);
        else ctx.lineTo(s.x, s.y);
      }});
      ctx.closePath();
      ctx.fillStyle = "#fff9ec";
      ctx.strokeStyle = "#d2c3a5";
      ctx.lineWidth = 1;
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }}

    function drawWalls(walls, toScreen, scale) {{
      ctx.save();
      ctx.lineCap = "round";
      for (const wall of walls) {{
        const a = point2(wall.startPoint || wall.start || wall.a);
        const b = point2(wall.endPoint || wall.end || wall.b);
        if (!a || !b) continue;
        const sa = toScreen(a.x, a.z);
        const sb = toScreen(b.x, b.z);
        ctx.strokeStyle = wall.color || "#555";
        ctx.lineWidth = Math.max(4, Number(wall.thickness || 0.12) * scale);
        line(sa.x, sa.y, sb.x, sb.y);
      }}
      ctx.restore();
    }}

    function drawGrid(bounds, scale, toScreen) {{
      ctx.strokeStyle = "#ece5d6";
      ctx.lineWidth = 1;
      for (let x = Math.floor(bounds.minX); x <= Math.ceil(bounds.maxX); x++) {{
        const a = toScreen(x, bounds.minZ);
        const b = toScreen(x, bounds.maxZ);
        line(a.x, a.y, b.x, b.y);
      }}
      for (let z = Math.floor(bounds.minZ); z <= Math.ceil(bounds.maxZ); z++) {{
        const a = toScreen(bounds.minX, z);
        const b = toScreen(bounds.maxX, z);
        line(a.x, a.y, b.x, b.y);
      }}
    }}

    function drawObject(item, index, toScreen, scale) {{
      const p = item.position || {{}};
      const s = item.size || [0.5, 0.5, 0.5];
      const c = toScreen(Number(p.x || 0), Number(p.z || 0));
      const w = Math.max(6, s[0] * scale);
      const h = Math.max(6, s[2] * scale);
      const yaw = yawFromQuat(item.rotation || {{ x: 0, y: 0, z: 0, w: 1 }});
      ctx.save();
      ctx.translate(c.x, c.y);
      ctx.rotate(-yaw);
      ctx.fillStyle = item.color || palette(index);
      ctx.strokeStyle = "#252525";
      ctx.lineWidth = 1.5;
      ctx.fillRect(-w / 2, -h / 2, w, h);
      ctx.strokeRect(-w / 2, -h / 2, w, h);
      ctx.fillStyle = "#111";
      ctx.font = "12px Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(index + 1), 0, 0);

      const arrowLength = Math.max(18, Math.min(w, h) * 0.55);
      ctx.strokeStyle = "#111";
      ctx.fillStyle = "#111";
      ctx.lineWidth = 2;
      line(0, 0, 0, -arrowLength);
      ctx.beginPath();
      ctx.moveTo(0, -arrowLength - 7);
      ctx.lineTo(-5, -arrowLength + 2);
      ctx.lineTo(5, -arrowLength + 2);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }}

    function drawOpening(item, toScreen, scale) {{
      const p = item.position || [0, 0, 0];
      const s = item.size || [0.8, 2, 0.1];
      const c = toScreen(Number(p[0] || 0), Number(p[2] || 0));
      ctx.save();
      ctx.translate(c.x, c.y);
      ctx.fillStyle = item.objectRole === "window" ? "#75b7ff" : "#d79a55";
      ctx.strokeStyle = "#333";
      ctx.lineWidth = 2;
      ctx.fillRect((-s[0] * scale) / 2, -4, Math.max(12, s[0] * scale), 8);
      ctx.strokeRect((-s[0] * scale) / 2, -4, Math.max(12, s[0] * scale), 8);
      ctx.fillStyle = "#111";
      ctx.font = "11px Arial";
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(item.objectRole || "opening", 0, -8);
      ctx.restore();
    }}

    function renderLegend(objects) {{
      legend.innerHTML = "";
      const note = document.createElement("div");
      note.className = "muted";
      note.style.margin = "8px 0 12px";
      note.textContent = "Mũi tên đen là hướng quay/front của đồ vật. Tường lấy từ test.json nếu result.json không có walls.";
      legend.appendChild(note);
      objects.forEach((item, index) => {{
        const row = document.createElement("div");
        row.className = "row";
        row.innerHTML = `<span class="swatch" style="background:${{item.color || palette(index)}}"></span><span>${{index + 1}}. ${{escapeHtml(item.name || "object")}}</span>`;
        legend.appendChild(row);
      }});
    }}

    function yawFromQuat(q) {{
      const x = Number(q.x || 0), y = Number(q.y || 0), z = Number(q.z || 0), w = Number(q.w ?? 1);
      return Math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + z * z));
    }}
    function line(x1, y1, x2, y2) {{
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }}
    function point2(value) {{
      if (Array.isArray(value)) return {{ x: Number(value[0] || 0), z: Number(value[1] || 0) }};
      if (value && typeof value === "object") return {{ x: Number(value.x || 0), z: Number(value.y ?? value.z ?? 0) }};
      return null;
    }}
    function palette(i) {{
      return ["#d8b98a", "#9fbf8f", "#94a7d6", "#d6988f", "#b894d6", "#88c2bd"][i % 6];
    }}
    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, c => ({{ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\\"": "&quot;", "'": "&#39;" }}[c]));
    }}
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
