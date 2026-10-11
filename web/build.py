import base64, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "parking1"
TEMPLATE_PATH = ROOT / "web" / "template.html"
ROUTING_PATH = ROOT / "web" / "routing.js"
OUTPUT_PATH = ROOT / "docs" / "index.html"

t = TEMPLATE_PATH.read_text(encoding="utf-8")
img = base64.b64encode((DATA_DIR / "img_0.jpg").read_bytes()).decode()
occ = json.loads((DATA_DIR / "occupancy.json").read_text(encoding="utf-8"))
slots = json.loads((DATA_DIR / "bounding_boxes.json").read_text(encoding="utf-8"))
t = (t.replace('__IMG__', 'data:image/jpeg;base64,' + img)
      .replace('__ROUTING__', ROUTING_PATH.read_text(encoding="utf-8"))
      .replace('__SLOTS__', json.dumps(slots, separators=(',',':')))
      .replace('__OCC__', json.dumps(occ['frames'], separators=(',',':')))
      .replace('__FPS__', str(occ['fps'])))
with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as output_file:
    output_file.write(t)
print(len(t)//1024, 'KB')
