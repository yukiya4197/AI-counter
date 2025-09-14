from ultralytics import YOLO
import cv2
from datetime import datetime

model = YOLO("yolo11n.pt")  
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

counted_ids = set()
hourly_counts = {}
daily_count = 0

previous_day_key = datetime.now().date()

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # 現在の時刻取得
    now = datetime.now()
    day_key = now.strftime("%Y-%m-%d")
    hour_key = now.strftime("%Y-%m-%d %H")

    # 日付が変わったらカウントをリセット
    if day_key != previous_day_key:
        daily_count = 0
        hourly_counts = {}
        previous_day_key = day_key

    # トラッキング付き推論（追跡モード）
    results = model.track(frame, persist=True)

    # 描画フレーム生成
    annotated_frame = results[0].plot()

    # トラッキングIDの取得
    boxes = results[0].boxes
    if boxes.id is not None and boxes.cls is not None:
        track_ids = boxes.id.cpu().numpy().astype(int).tolist()
        class_ids = boxes.cls.cpu().numpy().astype(int).tolist()
        for track_id, cls_id in zip(track_ids, class_ids):
            if model.names[int(cls_id)] == "person":
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    if hour_key not in hourly_counts:
                        hourly_counts[hour_key] = 0
                    hourly_counts[hour_key] += 1
                    daily_count += 1


    # 累計表示
    cv2.putText(annotated_frame, f"{day_key} Total Visitors: {daily_count}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 0), 2)

    cv2.imshow("Tracking Count", annotated_frame)


    if cv2.waitKey(1) & 0xFF in [27, ord('q')]: 
        break

cap.release()
cv2.destroyAllWindows()

