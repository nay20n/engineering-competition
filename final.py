# ===================================================================
# AI 토마토 자동 수확 로봇 시스템 - v4 (카메라 안정화 기능 추가)
# ===================================================================
# - 카메라 캡처 시 안정화 시간을 부여하고 해상도를 고정하여,
#   매번 일관되고 선명한 이미지를 얻도록 캡처 로직을 개선합니다.
# ===================================================================

import torch
import cv2
import serial
import time
import os
import sys

# 설정
YOLO_REPO_PATH = '/home/tomato/yolov5' # ⚠️ 중요: git clone한 yolov5 폴더 경로
MODEL_PATH = '/home/tomato/deeplearning/yolov5/best.pt'

# 아두이노 시리얼 포트
SERIAL_PORT = '/dev/ttyACM0' # 라즈베리파이에 연결된 아두이노 포트일 가능성이 높습니다.
# 시리얼 통신 속도
BAUD_RATE = 115200 # 아두이노 스케치와 동일한 속도로 설정

# --- 고급 설정 ---
CAM_INDEX = 0
FRAME_WIDTH = 640  # 캡처할 프레임 너비
FRAME_HEIGHT = 480 # 캡처할 프레임 높이
CONF_THRESHOLD = 0.9
IOU_THRESHOLD = 0.4
RESULT_IMAGE_PATH = "detection_result.png"

# 시리얼 통신 신호 정의
ARDUINO_CHECK_SIGNAL = "CHECK"
ARDUINO_END_SIGNAL = "END"
NO_TOMATO_SIGNAL = "None"
# -------------------------------------------------------------------

def initialize():
    """모델, 시리얼 포트를 모두 초기화합니다."""
    print("--- 시스템 초기화 시작 ---")
    model = None
    print("YOLOv5 모델을 로드하는 중입니다...")
    try:
        model = torch.hub.load(YOLO_REPO_PATH, 'custom', path=MODEL_PATH, source='local')
        model.conf = CONF_THRESHOLD
        model.iou = IOU_THRESHOLD
        print("모델 로드 완료.")
    except Exception as e:
        print(f"오류: YOLO 모델 로드에 실패했습니다. {e}")
        sys.exit()

    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=None)
        time.sleep(2)
        print(f"아두이노({SERIAL_PORT}) 연결 완료.")
    except serial.SerialException as e:
        print(f"❌ 오류: 시리얼 포트를 열 수 없습니다. {e}")
        sys.exit()
    
    print("--- 초기화 완료 ---")
    return model, ser

def preprocess_frame(frame):
    """
    캡처된 프레임의 품질을 향상시켜 탐지율을 높입니다.
    1. 자동 대비 조절 (CLAHE) - 더 자연스러운 대비 향상
    2. 노이즈 제거 (Bilateral Filter) - 경계선은 유지하며 노이즈 제거
    """
    # 1. 컬러 이미지를 LAB 색 공간으로 변환하여 L(밝기) 채널에만 적용
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    equalized_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 2. 노이즈 제거
    denoised_frame = cv2.bilateralFilter(equalized_frame, 9, 75, 75)
    
    return denoised_frame

def capture_and_detect(model):
    """[✨ 개선 사항] 안정화된 카메라 프레임을 캡처하고, 전처리 후, 객체를 탐지합니다."""
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"오류: 카메라(인덱스: {CAM_INDEX})를 열 수 없습니다.")
        return None, None
    
    try:
        # [✨ 개선 사항 1] 해상도 명시적 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print(f"카메라 해상도를 {FRAME_WIDTH}x{FRAME_HEIGHT}로 설정합니다.")
        
        # [✨ 개선 사항 2] 카메라 안정화 시간 부여
        # 자동 노출, 초점 등이 안정될 시간을 줍니다.
        time.sleep(1) 

        # [✨ 개선 사항 3] 버퍼 클리어링
        # 불안정한 초기 프레임을 버리기 위해 여러 번 읽어들입니다.
        for _ in range(10):
            ret, frame = cap.read()
            if not ret:
                print("오류: 카메라 안정화 중 프레임을 읽을 수 없습니다.")
                return None, None
        
        print("✅ 카메라 안정화 및 최종 프레임 캡처 완료.")

    finally:
        cap.release() # 작업이 끝나면 반드시 카메라 리소스를 해제합니다.
    
    if not ret or frame is None:
        print("오류: 최종 프레임 캡처에 실패했습니다.")
        return None, None
    
    # 이미지 전처리 단계
    print("이미지를 전처리하여 품질을 향상시킵니다...")
    preprocessed_frame = preprocess_frame(frame)

    print("이미지에서 객체 탐지를 수행합니다...")
    results = model(preprocessed_frame)
    
    return frame, results # 시각화는 원본 프레임(frame)에 하기 위해 함께 반환

# process_results, perform_scan_and_send, main 함수는 이전과 동일합니다.
def process_results(frame, results):
    """탐지 결과를 분석하고 원본 프레임에 시각화하며 좌표를 반환합니다."""
    detections = results.xyxy[0]
    coords_to_send = []
    
    colors = {0: (0, 255, 255), 1: (0, 165, 255), 2: (0, 0, 255), 3: (255, 0, 255)}

    for *box, conf, cls_idx_tensor in detections:
        x1, y1, x2, y2 = map(int, box)
        cls_idx = int(cls_idx_tensor)
        
        if cls_idx not in colors: continue
            
        xc = (x1 + x2) // 2
        yc = (y1 + y2) // 2
        color = colors[cls_idx]
        
        print(f"  → (신뢰도: {conf:.2f}), 중심: ({xc}, {yc})")
        coords_to_send.append(f"{xc},{yc}")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.imwrite(RESULT_IMAGE_PATH, frame)
    print(f"🖼탐지 결과가 '{RESULT_IMAGE_PATH}' 파일로 저장되었습니다.")
    return coords_to_send

def perform_scan_and_send(model, ser):
    """스캔, 탐지, 결과 분석, 데이터 전송까지의 한 사이클을 수행합니다."""
    original_frame, results = capture_and_detect(model)
    if original_frame is None or results is None:
        print("이미지 처리 오류. 작업을 건너뜁니다.")
        return

    coords_to_send = process_results(original_frame, results)

    if coords_to_send:
        data_str = '/'.join(coords_to_send) + '/\n'
        ser.write(data_str.encode('utf-8'))
        print(f"좌표 전송: {data_str.strip()}")
    else:
        data_str = NO_TOMATO_SIGNAL + '\n'
        ser.write(data_str.encode('utf-8'))
        print(f"탐지된 토마토 없음: '{NO_TOMATO_SIGNAL}' 신호 전송")

def main():
    """메인 실행 함수 (첫 스캔은 자동, 이후는 요청-응답 방식)"""
    model, ser = initialize()

    try:
        print("\n프로그램 시작! 첫 번째 스캔을 자동으로 수행합니다.")
        perform_scan_and_send(model, ser)

        while True:
            print(f"\n아두이노로부터 '{ARDUINO_CHECK_SIGNAL}' 또는 '{ARDUINO_END_SIGNAL}' 신호 수신 대기 중...")
            line = ser.readline().decode('utf-8').strip()

            if line == ARDUINO_CHECK_SIGNAL:
                print(f"✅ '{line}' 신호 수신! 다음 스캔을 시작합니다.")
                perform_scan_and_send(model, ser)

            elif line == ARDUINO_END_SIGNAL:
                print(f"✅ '{line}' 신호 수신! 프로그램을 종료합니다.")
                break
            
            else:
                if line:
                    print(f"⚠️ 알 수 없는 신호 수신: '{line}'. 무시하고 대기합니다.")

    except KeyboardInterrupt:
        print("\n사용자에 의해 프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("시리얼 포트 연결을 닫았습니다.")
        print("--- 프로그램 종료 ---")

if __name__ == '__main__':
    main()



