import argparse, json, cv2, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "parking1"
VIDEO_PATH = ROOT / "outputs" / "output.mp4"

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", nargs="?", type=Path, default=DATA_DIR)
parser.add_argument("video_path", nargs="?", type=Path, default=VIDEO_PATH)
args = parser.parse_args()

zones = json.loads((args.data_dir / "bounding_boxes.json").read_text(encoding="utf-8"))
cap = cv2.VideoCapture(str(args.video_path))
fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W,H = int(cap.get(3)), int(cap.get(4))
print("output.mp4", W, H, fps, n)
masks=[]
for z in zones:
    m = np.zeros((H,W),np.uint8)
    cv2.polylines(m,[np.array(z["points"],np.int32).reshape(-1,1,2)],True,255,3)
    masks.append(m>0)
frames=[]; i=0
while True:
    ok,f = cap.read()
    if not ok: break
    f = f.astype(int); b,g,r = f[...,0],f[...,1],f[...,2]
    red = (r>180)&(g<90)&(b<90); green=(g>180)&(r<90)&(b<90)
    row=[]
    for m in masks:
        rc,gc = int((red&m).sum()), int((green&m).sum())
        row.append(1 if rc>gc else 0)   # 1 = occupied
    frames.append(row); i+=1
print("frames",i)
A=np.array(frames); print("occupied per slot (fraction):", np.round(A.mean(0),2).tolist())
print("changes in total occupied:", sorted(set(A.sum(1).tolist())))
with (args.data_dir / "occupancy.json").open("w", encoding="utf-8") as occupancy_file:
    json.dump({"fps":round(fps,2),"frames":frames},occupancy_file,separators=(",",":"))
