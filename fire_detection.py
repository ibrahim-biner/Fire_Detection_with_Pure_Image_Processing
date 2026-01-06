import cv2
import numpy as np


video_path = "yangin_test.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise SystemExit("Video acilamadi.")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 1 or fps != fps:
    fps = 25.0
delay_ms = int(1000 / fps)

print("ESC: cikis | SPACE: dur/devam")
print("+/-: hareket esigi (th_m)")
print("W/S: H esigi | E/D: S esigi | R/F: V esigi")
print("]/[: min_area | O/P: fire_area_th | K/L: confirm_frames")
print("M: Hareket modu (MOG2/FrameDiff)")

# Kontrol ve kullanıcı arayüzü ayarları 
paused = False


use_mog2 = True  
fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=50, detectShadows=True)
mog2_learning_rate = 0.001  


th_m = 21

# --- HSV parametreleri: Alev rengini tanımlamak için ---
# H (Hue): 0-35 (Kırmızı-Sarı-Turuncu)
# S (Saturation): 60-255 (Beyaz ışık eleme - S düşükse beyaz/spot ışık olabilir)
# V (Value): 135-255 (Parlaklık eşiği)
th_h_low = 0
th_h_high = 35
th_s_low = 60   # Beyaz ışık S<60 olduğu için elenir
th_s_high = 255
th_v_low = 135
th_v_high = 255


min_area = 120

# Zamansal dogrulama 
frame_fire_count = 0
fire_confirm_frames = 10
fire_area_th = 250
fire_confirmed = False


alarm_saved = False

prev_gray = None
kernel3 = np.ones((3, 3), np.uint8)
kernel5 = np.ones((5, 5), np.uint8)
kernel15 = np.ones((15, 15), np.uint8)  # Delik doldurma 

# 3-kanal BGR ye çevriliyor 
def to_bgr(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def resize_to(img, w, h):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        # 1) HAREKET MASKESİ 
        if use_mog2:
            
            mog_mask = fgbg.apply(frame, learningRate=mog2_learning_rate)
            
            # Gölgeleri temizle adımı
            _, motion = cv2.threshold(mog_mask, 250, 255, cv2.THRESH_BINARY)
            
            # Gürültü temizleme adımı
            motion = cv2.erode(motion, kernel3, iterations=1)
            motion = cv2.dilate(motion, kernel3, iterations=2)
        else:
            # Frame Differencing ile hareket algılama 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_gray is None:
                prev_gray = gray
                continue

            diff = cv2.absdiff(gray, prev_gray)
            _, motion = cv2.threshold(diff, th_m, 255, cv2.THRESH_BINARY)
            motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, kernel3, iterations=1)
            motion = cv2.dilate(motion, kernel5, iterations=1)
            prev_gray = gray

        #  ALEV RENGİ MASKESİ 
        # Sadece hareket eden bölgeye odaklanıyoruz (optimizasyon)
        moving_object = cv2.bitwise_and(frame, frame, mask=motion)
        hsv = cv2.cvtColor(moving_object, cv2.COLOR_BGR2HSV)

        # HSV eşikleme (S>60 ile beyaz ışık eleme)
        lower_fire = np.array([th_h_low, th_s_low, th_v_low])
        upper_fire = np.array([th_h_high, th_s_high, th_v_high])
        fire_color = cv2.inRange(hsv, lower_fire, upper_fire)

        #  Delik Doldurma 
        # Ateşin beyaz merkezindeki delikleri kapatıyoruz
        fire_color = cv2.morphologyEx(fire_color, cv2.MORPH_CLOSE, kernel15, iterations=1)
        fire_color = cv2.dilate(fire_color, kernel3, iterations=2)

        # 3) YANGIN ADAYI = Renk ve Hareket birleşimi
        fire_candidate = cv2.bitwise_and(fire_color, motion)
        fire_candidate = cv2.morphologyEx(fire_candidate, cv2.MORPH_CLOSE, kernel5, iterations=2)

        # 4) ZAMANSAL DOĞRULAMA 
        fire_area = cv2.countNonZero(fire_color)
        candidate_area = cv2.countNonZero(fire_candidate)

        if (fire_area > fire_area_th) and (candidate_area > 30):
            frame_fire_count += 1
        else:
            frame_fire_count = max(0, frame_fire_count - 1)

        if frame_fire_count >= fire_confirm_frames:
            fire_confirmed = True

        #  5) KUTULAR 
        vis = frame.copy()
        contours, _ = cv2.findContours(fire_candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            aspect = w / float(h + 1e-6)
            if aspect > 4.0 or aspect < 0.2:
                continue
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(vis, "FIRE?", (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 6) DURUM YAZISI
        mode_text = "MOG2" if use_mog2 else "FrameDiff"
        cv2.putText(vis, f"Mode: {mode_text} | min_area={min_area}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"HSV: H:{th_h_low}-{th_h_high} S:{th_s_low}-{th_s_high} V:{th_v_low}-{th_v_high} | area={fire_area}", 
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if fire_confirmed:
            cv2.putText(vis, "FIRE DETECTED", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        else:
            cv2.putText(vis, f"counter: {frame_fire_count}/{fire_confirm_frames}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # 7) ALARM KAYDI 
        if fire_confirmed and not alarm_saved:
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t_sec = t_ms / 1000.0

            print(f"FIRE DETECTED at ~{t_sec:.2f} sec")
            cv2.imwrite("fire_detected_frame.png", vis)

            with open("fire_log.txt", "a", encoding="utf-8") as f:
                f.write(f"FIRE DETECTED at {t_sec:.2f} sec\n")

            alarm_saved = True

        #  8) DASHBOARD 
        tile_w, tile_h = 640, 360
        a = resize_to(vis, tile_w, tile_h)
        b = resize_to(to_bgr(fire_color), tile_w, tile_h)
        c = resize_to(to_bgr(motion), tile_w, tile_h)
        d = resize_to(to_bgr(fire_candidate), tile_w, tile_h)

        cv2.putText(a, "Video + Kutular", (10, tile_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(b, "Alev rengi maskesi (HSV + Delik Doldurma)", (10, tile_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(c, f"Hareket maskesi ({mode_text})", (10, tile_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(d, "Yangin adayi = renk AND hareket", (10, tile_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        dashboard = cv2.vconcat([cv2.hconcat([a, b]), cv2.hconcat([c, d])])
        cv2.imshow("Hibrit Yangin Tespit - MOG2 + HSV + Zamansal", dashboard)

    # Klavye kontrolleri
    key = cv2.waitKey(delay_ms) & 0xFF
    if key == 27:
        break
    if key == 32:
        paused = not paused

    
    if key == ord('m'):
        use_mog2 = not use_mog2
        if use_mog2:
            print("Hareket modu: MOG2 (Gelismis)")
        else:
            print("Hareket modu: Frame Differencing (Hizli)")

    
    if key in (ord('+'), ord('=')):
        th_m = min(80, th_m + 2)
    if key in (ord('-'), ord('_')):
        th_m = max(5, th_m - 2)

    
    # W/S: Hue high
    if key == ord('w'):
        th_h_high = min(179, th_h_high + 5)
    if key == ord('s'):
        th_h_high = max(th_h_low + 1, th_h_high - 5)
    
    # E/D: Saturation low 
    if key == ord('e'):
        th_s_low = min(th_s_high - 1, th_s_low + 5)
    if key == ord('d'):
        th_s_low = max(0, th_s_low - 5)
    
    # R/F: Value low
    if key == ord('r'):
        th_v_low = min(th_v_high - 1, th_v_low + 5)
    if key == ord('f'):
        th_v_low = max(0, th_v_low - 5)

    # Alan filtreleri
    if key == ord(']'):
        min_area = min(5000, min_area + 50)
    if key == ord('['):
        min_area = max(50, min_area - 50)

    # Zamansal parametreler
    if key == ord('o'):
        fire_area_th = max(50, fire_area_th - 50)
    if key == ord('p'):
        fire_area_th = min(50000, fire_area_th + 50)
    if key == ord('k'):
        fire_confirm_frames = max(3, fire_confirm_frames - 1)
    if key == ord('l'):
        fire_confirm_frames = min(120, fire_confirm_frames + 1)

cap.release()
cv2.destroyAllWindows()
print("Bitti.")