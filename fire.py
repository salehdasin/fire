import cv2
import torch
import numpy as np

# بارگذاری مدل
model = torch.hub.load('.', 'custom', path='yolov5s.pt', source='local')

# تنظیم دوربین (یا مسیر فایل ویدیو)
cap = cv2.VideoCapture(0)  # برای وبکم، یا مسیر فایل ویدیو را وارد کنید

def is_fire_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # محدوده رنگ قرمز (آتش)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    # محدوده رنگ نارنجی
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_orange, upper_orange)
    
    mask = mask1 + mask2 + mask3
    
    return cv2.countNonZero(mask) > 0.05 * roi.size

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # پیش‌پردازش تصویر
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)

    # انجام تشخیص
    results = model(frame)

    # پردازش و نمایش نتایج
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if conf > 0.2:  # کاهش آستانه اطمینان
            roi = frame[int(y1):int(y2), int(x1):int(x2)]
            if is_fire_color(roi):
                label = f"Potential Fire: {conf:.2f}"
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                print("احتمال وجود آتش! هشدار!")

    # نمایش تصویر
    cv2.imshow('Fire Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
