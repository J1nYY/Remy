# 전처리 및 후처리 추가
import cv2, torch, pandas, numpy as np, threading, queue, ncnn, sounddevice as sd, scipy.io.wavfile as wav
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.vision import RunningMode
import RPi.GPIO as GPIO
import time,math
from collections import deque
# ===== ncnn =====
net = ncnn.Net()
net.opt.num_threads = 4
net.opt.use_fp16_storage = True
net.opt.use_fp16_arithmetic = True
net.load_param("kitchen_tools_best03_50.ncnn.param")
net.load_model("kitchen_tools_best03_50.ncnn.bin")
DEBUG_DRAW=True
IMG_SIZE = 320
CONF_TH = 0.6
FRAME_SKIP = 1
IOU_TH = 0.25
DETECT_PADDING = 30

WARNING_TXT_ORG = (30,30)
DERECTION_ORG = (430,450)
TARGET_TXT_ORG=(30,65)
COLOR_RED = (0,0,255)
COLOR_GREEN = (0,255,0)
COLOR_BLUE = (255,0,0)
TARGET=0


LED_NUMBER = [2,3,4,14,15,18,17,27,22]
TOOLS_NAME = ["Knife", "Fork", "Ladle", "Plate"]
CHUNK=512

fid = 0
target_pixel = []
target_pixel_ttl=0
blade_pixel = []
blade_pixel_ttl=0
flag_target=False
flag_blade=False
# ===== 사운드 로드 (float32 변환 + SR 맞추기) =====
def _to_float32(w):
    return w.astype(np.float32) / np.iinfo(w.dtype).max if w.dtype != np.float32 else w

def _resample_linear(x, sr_from, sr_to):
    if sr_from == sr_to:
        return x
    # 간단 선형보간(지연 적고 의존성 없음)
    r = sr_to / sr_from
    idx = np.arange(0, len(x)*r, 1.0)
    xp  = np.arange(len(x))
    # 경계 보호
    idx_clamped = np.clip(idx, 0, len(x)-1)
    x0 = np.floor(idx_clamped).astype(np.int32)
    x1 = np.minimum(x0+1, len(x)-1)
    frac = idx_clamped - x0
    return ((1-frac)*x[x0] + frac*x[x1]).astype(np.float32)

# 위험음
DANGER_SR, DANGER_WAV = wav.read("danger.wav")
DANGER_WAV = _to_float32(DANGER_WAV)
# 감지음
DETECT_SR, DETECT_WAV = wav.read("detect.wav")
DETECT_WAV = _to_float32(DETECT_WAV)

# 재생 샘플레이트 기준(위험음 SR로 통일)
PLAY_SR = DANGER_SR
DETECT_WAV = _resample_linear(DETECT_WAV, DETECT_SR, PLAY_SR)

# 모노 보장(스테레오면 1ch로 다운믹스)
if DANGER_WAV.ndim > 1: DANGER_WAV = DANGER_WAV.mean(axis=1)
if DETECT_WAV.ndim > 1: DETECT_WAV = DETECT_WAV.mean(axis=1)


sound_state = {
    "mode": None,           # None | "danger" | "detected"
    "idx": 0,               # 현재 재생 위치(샘플 인덱스)
    "oneshot": False,       # True면 샘플 한 번만 재생
    "playing": False,       # 콜백 내부에서 사용
}

def set_hold_mode(event):
    """이벤트가 True면 'danger'/'detected', False면 None로 호출"""
    # hold 모드: 루프 재생. 켤 때만 idx 리셋(원하면 유지해도 됨)
    if event is None:
        sound_state["mode"] = None
        sound_state["playing"] = False
    else:
        if sound_state["mode"] != event:
            sound_state["idx"] = 0
        sound_state["mode"] = event
        sound_state["oneshot"] = False
        sound_state["playing"] = True

def pulse_once(event):
    """엣지에서 1회만(2초짜리 전체) 재생. 중첩 없음."""
    # 이미 같은 oneshot이 진행 중이면 무시 → 중첩 방지
    if sound_state["playing"] and sound_state["oneshot"] and sound_state["mode"] == event:
        return
    sound_state["mode"] = event
    sound_state["idx"] = 0
    sound_state["oneshot"] = True
    sound_state["playing"] = True
# ===== 오디오 콜백/큐 =====


def audio_cb(outdata, frames, timeinfo, status):
    # outdata: shape (frames, channels)  => channels=1이면 (frames, 1)
    m = sound_state["mode"]

    # ---- 빠른 무음 채우기 (shape 일치) ----
    if not sound_state["playing"] or m is None:
        # 무조건 0.0으로 채우면 모양 문제 없음
        outdata.fill(0.0)
        return

    if m == "danger":
        src = DANGER_WAV  # 1D float32
    elif m == "detected":
        src = DETECT_WAV
    else:
        outdata.fill(0.0)
        return

    # 안전가드: dtype / C-연속 보장
    if src.dtype != np.float32:
        src = src.astype(np.float32, copy=False)
    src = np.ascontiguousarray(src)

    n = len(src)
    i = sound_state["idx"]
    j = i + frames          # ← 꼭 frames 사용

    # outdata는 (frames,1) 이므로 1D 소스 -> 2D 타겟 복사
    if j <= n:
        # 연속 구간
        chunk = src[i:j]
        outdata[:, 0] = chunk
    else:
        # 래핑(루프)
        k = j - n
        # 뒤쪽 조각
        tail = src[i:n]        # 길이 n-i
        # 앞쪽 조각
        head = src[:k]         # 길이 k
        # 길이 체크(방어)
        # 만약 frames > n 이면 head/tail로 다 못채울 수 있으니 min으로 자르기
        tlen = min(n - i, frames)
        outdata[:tlen, 0] = tail[:tlen]
        if tlen < frames:
            hlen = min(k, frames - tlen)
            outdata[tlen:tlen+hlen, 0] = head[:hlen]
        # 나머지는 무음(이론상 없어야 하지만 방어)
        if tlen + hlen < frames:
            outdata[tlen+hlen:, 0].fill(0.0)

    sound_state["idx"] = j % n

    # oneshot 끝나면 멈춤
    if sound_state["oneshot"] and sound_state["idx"] == 0:
        sound_state["playing"] = False
        sound_state["mode"] = None

# ===== 그리기 =====
def draw_landmarks_on_image(tools_dect, hand_dect, frame):
    landmarks_list = hand_dect.hand_landmarks
    annotated_image = np.copy(frame)

    # 도구 박스
    for (x1,y1,x2,y2), s, _ in tools_dect:
        cv2.rectangle(annotated_image, (x1,y1), (x2,y2), (0,0,255), 2)

    # 손 랜드마크
    for landmark in landmarks_list:
        landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        for lm in landmark:
            landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)])
        if DEBUG_DRAW:
            solutions.drawing_utils.draw_landmarks(annotated_image,landmarks_proto,solutions.hands.HAND_CONNECTIONS,solutions.drawing_styles.get_default_hand_landmarks_style())
    draw_text(annotated_image,TOOLS_NAME[TARGET], TARGET_TXT_ORG,(255,255,0))
    return annotated_image
def setting_target(tools_result):
    global TOOLS_NAME, TARGET, target_pixel,target_pixel_ttl,fid
    if TOOLS_NAME[TARGET] == "Knife": target = 2
    elif TOOLS_NAME[TARGET] == "Fork": target = 1
    elif TOOLS_NAME[TARGET] == "Ladle": target = 3
    elif TOOLS_NAME[TARGET] == "Plate": target = 4
    else: target = 2
    target_xy = [ tools[0] for tools in tools_result if tools[2] == target ]
    if len(target_xy) > 0:
        target_pixel = save_pixel(target_xy)
        target_pixel_ttl=fid
def inside_allhand(hand_pixel,target_pixel,W,H):
    for lm in hand_pixel:
        lm_x = int(lm.x * W); lm_y = int(lm.y * H)
        for target in target_pixel:
            if check_inside(target, lm_x, lm_y) :
                return True
    return False
# ===== 겹칩 확인 + 오디오 이벤트 리턴 =====
def detection_box(tools_dect, hand_dect, frame):
    blade_xy  = [ tools[0] for tools in tools_dect if tools[2] == 0 ]
    setting_target(tools_dect)

    global target_pixel, blade_pixel,flag_target,flag_blade,blade_pixel_ttl,fid
    if len(blade_xy)  > 0:
        blade_pixel  = save_pixel(blade_xy)
        blade_pixel_ttl=fid
    annotated_image = np.copy(frame)
    hand_list = hand_dect.hand_landmarks
    H, W, _ = annotated_image.shape


    flag_Up = False; flag_Down = False; flag_Left = False; flag_Right = False
    
    if not hand_list:
        flag_blade=False
        flag_target=False
    hand_pixel=[]
    for hand in hand_list:
        middle_x = (int(hand[8].x*W))
        middle_y = (int(hand[8].y*H))
        cv2.circle(annotated_image, (middle_x, middle_y), 5, COLOR_BLUE, -1, cv2.LINE_AA)
        for lm in hand:
            hand_pixel.append(lm)
        # 손잡이 확인
        for target in target_pixel:
            target_middle_x = (target[0] + target[2]) // 2
            target_middle_y = (target[1] + target[3]) // 2
            cv2.circle(annotated_image, (target_middle_x, target_middle_y), 5, (255,255,0), -1, cv2.LINE_AA)

            
            if middle_x < target_middle_x-DETECT_PADDING: flag_Right = True
            elif middle_x > target_middle_x+DETECT_PADDING: flag_Left = True
            if middle_y < target_middle_y-DETECT_PADDING:  flag_Down  = True
            elif middle_y > target_middle_y+DETECT_PADDING: flag_Up    = True

        # 날 확인

    flag_target=inside_allhand(hand_pixel,target_pixel,W,H)
    flag_blade=inside_allhand(hand_pixel,blade_pixel,W,H)
    false_LED()
    if flag_Left:
        if flag_Up:    GPIO.output(22, True)
        elif flag_Down:GPIO.output(17, True)
        else:          GPIO.output(27, True)
    elif flag_Right:
        if flag_Up:    GPIO.output(4, True)
        elif flag_Down:GPIO.output(2, True)
        else:          GPIO.output(3, True)
    else:
        if flag_Up:    GPIO.output(18, True)
        elif flag_Down:GPIO.output(14, True)

    audio_event = ["None"]
    if flag_blade:
        draw_text(annotated_image, "DANGER", WARNING_TXT_ORG, COLOR_RED)
        audio_event[0]="danger" 
        GPIO.output(15, True)
        print(f"danger {audio_event}")
                #  여기서 danger.wav 재생
    elif flag_target:
        draw_text(annotated_image, "DETECTED", WARNING_TXT_ORG, COLOR_GREEN)
        audio_event[0]="detected"
        detected_LED()
        print(f"detect {audio_event}")
    #print(f"target:{flag_target},blade:{flag_blade}")

    return annotated_image, audio_event[0]

def save_pixel(boxes):
    return [[x1,y1,x2,y2] for x1,y1,x2,y2 in boxes]

def check_inside(box, x, y):
    return (box[0] <= x <= box[2]) and (box[1] <= y <= box[3])

def draw_text(image, text, org, color):
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)

def false_LED():
    global LED_NUMBER
    for led in LED_NUMBER:
        GPIO.output(led, False)

def detected_LED():
    global LED_NUMBER
    for led in LED_NUMBER:
        GPIO.output(led, True)
def button_callback(channel):
    global TARGET, TOOLS_NAME,target_pixel
    if TARGET < len(TOOLS_NAME):
        TARGET += 1
    if TARGET >= len(TOOLS_NAME):
        TARGET = 0
    target_pixel=[]
# ===== 전처리 =====
def letterbox(img, new=IMG_SIZE, color=(114,114,114)):
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((new, new, 3), color, dtype=np.uint8)
    top = (new - nh) // 2
    left = (new - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, r, left, top

def nms(dets, iou_th=IOU_TH):
    if not dets: return []
    boxes  = np.array([d[0] for d in dets], dtype=np.float32)
    scores = np.array([d[1] for d in dets], dtype=np.float32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        rest = order[1:]
        xx1 = np.maximum(boxes[i,0], boxes[rest,0])
        yy1 = np.maximum(boxes[i,1], boxes[rest,1])
        xx2 = np.minimum(boxes[i,2], boxes[rest,2])
        yy2 = np.minimum(boxes[i,3], boxes[rest,3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i,2] - boxes[i,0]) * (boxes[i,3] - boxes[i,1])
        area_r = (boxes[rest,2] - boxes[rest,0]) * (boxes[rest,3] - boxes[rest,1])
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = rest[iou <= iou_th]
    return [dets[k] for k in keep]

def tools_inference(frame_rgb):
    H, W, _ = frame_rgb.shape            # (버그 수정) frame -> frame_rgb
    img_lbx, r, lpad, tpad = letterbox(frame_rgb, IMG_SIZE)
    # (버그 수정) img_rgb → img_lbx는 이미 RGB
    input_mat = ncnn.Mat.from_pixels(img_lbx, ncnn.Mat.PixelType.PIXEL_RGB, IMG_SIZE, IMG_SIZE)
    input_mat.substract_mean_normalize([0,0,0], [1/255.0, 1/255.0, 1/255.0])

    ex = net.create_extractor()
    ex.input("in0", input_mat)
    _, result = ex.extract("out0")

    arr = result.numpy()
    D, N = arr.shape
    A = arr.T
    
    cx, cy, w, h = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
    cls_scores = A[:, 4:]
    
    cls_ids = np.argmax(cls_scores, axis=1)
    scores  = cls_scores[np.arange(A.shape[0]), cls_ids]
    
    keep = scores >= CONF_TH
    if not np.any(keep): return []
    cx = cx[keep]; cy = cy[keep]; w = w[keep]; h = h[keep]
    scores = scores[keep]; cls_ids = cls_ids[keep]

    x1 = cx - w/2; y1 = cy - h/2
    x2 = cx + w/2; y2 = cy + h/2
    x1 = (x1 - lpad) / r; y1 = (y1 - tpad) / r
    x2 = (x2 - lpad) / r; y2 = (y2 - tpad) / r

    x1 = np.clip(x1, 0, W); y1 = np.clip(y1, 0, H)
    x2 = np.clip(x2, 0, W); y2 = np.clip(y2, 0, H)

    dets = [([int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])],
             float(scores[i]), int(cls_ids[i])) for i in range(len(scores))]
    dets = nms(dets, IOU_TH)
    return dets

# ===== 메인 =====
def main():
    # 손 랜드마크
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2,running_mode=RunningMode.VIDEO )
    hand_detector = vision.HandLandmarker.create_from_options(options)

    # 캠
    cap = cv2.VideoCapture(0)
    global fid,blade_pixel_ttl,blade_pixel,target_pixel,target_pixel_ttl

    if not cap.isOpened():
        print("웹 캠을 열 수 없습니다")

    GPIO.setmode(GPIO.BCM)
    for led in LED_NUMBER:
        GPIO.setup(led,GPIO.OUT)
    GPIO.setup(21, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.add_event_detect(21, GPIO.RISING, callback=button_callback, bouncetime=300)
    print(sd.query_devices())

   # 최근 30프레임 평균
    t_prev = time.time()
    fid_log=0
    disp_fps = 0.0
    # === 오디오 스트림: 콜백 방식 (블로킹 없음) ===
    with sd.OutputStream(channels=1, samplerate=PLAY_SR, dtype="float32",
                         blocksize=CHUNK, callback=audio_cb):
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            fid += 1
            if fid % (FRAME_SKIP + 1) == 1:
                tools_result = tools_inference(frame_rgb)
                ts_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
                hand_result = hand_detector.detect_for_video(mp_image, ts_ms)
            
            drawing_image = draw_landmarks_on_image(tools_result, hand_result, frame)
            annotated_image, audio_event = detection_box(tools_result, hand_result, drawing_image)
            if blade_pixel and fid-blade_pixel_ttl>5 and audio_event=='None':
                blade_pixel=[]
            if target_pixel and fid-target_pixel_ttl>20 and audio_event=='None':
                target_pixel=[]
            t_now = time.time()
            elased_time=t_now-t_prev
            if elased_time>1.0:
                print(audio_event=="detected")
                t_prev+=elased_time
            if audio_event == "danger":
                set_hold_mode("danger")
            elif audio_event == "detected":
                set_hold_mode("detected")
            else:
                set_hold_mode(None)
            cv2.imshow("hand_land", annotated_image)
            # === FPS 갱신 ===
            '''
            t_now = time.time()
            elased_time=t_now-t_prev
            if elased_time>1.0:
                #print(f"fps:{((fid-fid_log)/elased_time):5.2f}")
                fid_log=fid
                t_prev+=elased_time
            '''
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    GPIO.cleanup()

if __name__ == "__main__":
    main()
