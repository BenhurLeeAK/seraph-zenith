# 10 m / 15 m field test -- do in this order

## A. PI, before going out (2 min)
```
hailortcli parse-hef ~/sar_yolov8s_cfb36.hef | grep -i "number of classes"     # must say 1
sha256sum ~/sar_yolov8s_cfb36.hef                                              # must start cfb36ef8
```
Do not use `~/yolov8s.hef` (6-class model, 8a4909c2...).
Use the same capture script and camera settings as the 3/5/7 m set (1456x1088, JPEG, no resize).

## B. In the field
- Camera mounted and pointed as in the 3/5/7 m set. Write down: camera height (m), tilt angle (deg), date, weather/light.
- Measure 10 m and 15 m with a tape from the point on the ground under the camera to the person's feet. Mark the spots.
- 15-20 frames per distance, 2-3 people per frame. Between frames CHANGE something: who stands where, facing (front/side/back),
  standing / crouching, background. No two frames the same.
- Session 1 and, if you can, session 2 (different time of day or different background). Put the session in the name.
- Names (the 10/15 is read from the name): `10m_s1_001.jpg`, `10m_s1_002.jpg`, ... `15m_s2_001.jpg`
- Keep a note: which people, which distance, per frame if it changes.

## C. Label (laptop)
Roboflow, class `person` or `human`, whole-body box incl. head and feet, YOLOv8 export. One class only.
Put the export in `eval_distance/` as `images/` and `labels/`. Never inside `data/processed/`.
Check: `ls eval_distance/images | wc -l` equals `ls eval_distance/labels | wc -l`. Roboflow adds a suffix to file names
(`..._jpg.rf.<hash>.jpg`); that is fine, `10m_s1_001` stays at the start.

## D. PI: run the model on the raw frames (use the unlabelled originals or the export; same pixels)
LAPTOP -> PI:
```
scp -r eval_distance/images seraphzenith@10.112.116.1:~/field_imgs
```
PI:
```
python hailo_predict.py --hef ~/sar_yolov8s_cfb36.hef --images ~/field_imgs --out ~/dets_hef_field.json --mode full
```
It saves boxes + scores + the HEF sha256 in the JSON. Keep the raw frames.

## E. LAPTOP: score (conf 0.30, IoU 0.5, no tuning)
```
cd ~/Desktop/SIH26177/AI/sar-human-detection
scp seraphzenith@10.112.116.1:~/dets_hef_field.json out/
python paper_eval.py distance --dets out/dets_hef_field.json --images eval_distance/images --labels eval_distance/labels \
    --out-dir out/dist_field --label hef_field --conf 0.3
python draw_check.py --dets out/dets_hef_field.json --images eval_distance/images --labels eval_distance/labels \
    --per-image out/dist_field/per_image_hef_field.csv --out out/draw_field --conf 0.3
# float model on the same frames (secondary):
python paper_eval.py distance --weights runs/detect/models/sar_stage2-2/weights/best.pt \
    --images eval_distance/images --labels eval_distance/labels --out-dir out/dist_field --label float_field --conf 0.3
```
Send back: the printed table, `distance_hef_field.csv`, `per_image_hef_field.csv`, `out/draw_field/*.jpg`, and the camera height/tilt/date.

Columns added: `persons_split_2plus_boxes` (people covered by 2+ boxes at IoU >= 0.1: legs vs torso, duplicates) and a per-image CSV.
Example images are chosen by rule (best and worst share of people found per distance), not by hand.
If a distance is poor, report it as is. Keep conf 0.30.
