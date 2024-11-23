import cv2
import numpy as np


# 이미지 불러오기
def load_image(file_path):
    """
    주어진 경로에서 이미지를 불러옴
    
    Args:
        file_path : 이미지 파일 경로
    """
    image = cv2.imread(file_path)
    if image is None:
        raise FileNotFoundError(f"File not found: {file_path}")
    return image


# 이미지 저장
def save_image(image, save_path):
    """주어진 경로에 이미지를 저장함

    Args:
        image : 저장할 이미지 객체
        save_path : 저장할 파일 경로
    """
    cv2.imwrite(save_path, image)


# 이미지 크기 조정
def resize_image(image, scale):
    """
    이미지를 주어진 배율로 크기를 조정함
    
    Args:
        image : 입력 이미지
        scale : 크기 조정 배율(float, 0.1~2.0)
    return : 크기 조정된 이미지
    """
    if scale <= 0:
        raise ValueError("Scale must be greater then 0.")
    
    height, width = image.shape[:2]
    new_width = int(width*scale)
    new_height = int(height*scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image


# 슬라이더 콜백 함수
def on_scale_change(val):
    global img, window_name
    scale = val / 100  # 슬라이더 값(0~200)을 0.01~2.00으로 변환
    resized = resize_image(img, scale)
    cv2.imshow(window_name, resized)
    

# 이미지 회전
def rotate_image(image, angle):
    """
    이미지를 주어진 각도(angle)로 회전함
    
    Args:
        image : 입력 이미지
        angle : 회전 각도(90, 180, 270 중 하나)
    return : 회전된 이미지
    """
    if angle not in [90, 180, 270]:
        raise ValueError("Angle must be 90, 180, or 270.")
    
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


# 이미지 반전
def flip_image(image, flip_code):
    """
    이미지를 좌우 또는 상하로 반전함
    
    Args:
        image : 입력 이미지
        filp_code: 1(좌우 반전), 0(상하 반전)
    return: 반전된 이미지
    """
    if flip_code not in [0, 1]:
        raise ValueError("Filp code must be 0(vertical) or 1(horizontal).")
    return cv2.flip(image, flip_code)


# 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    """
    마우스 클릭 이벤트를 처리함
    Args:
        event : 마우스 이벤트 타입(ex. 좌클릭, 우클릭 등)
        x : 클릭한 x좌표
        y : 클릭한 y 좌표
        flags : 이벤트 플래그
        param : 추가 매개변수
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        color = param[y, x]  # 클릭한 좌표의 색상 정보 BGR 형식
        print(f"Mouse clicked at ({x}, {y}) with color: {color}")


# ROI 선택 함수
def select_roi(image):
    """
    드래그하여 ROI를 선택함
    Args:
        image : 입력된 이미지

    return: 선택된 ROI
    """
    roi = cv2.selectROI("Select ROI", image, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi
    if w == 0 or h == 0:
        raise ValueError("ROI not selected or has zero size.")
    return image[int(y):int(y+h), int(x):int(x+w)]


# 그레이스케일 변환
def convert_to_grayscale(image):
    """
    입력 이미지를 그레이스케일로 변환

    Args:
        image : 입력 이미지

    return: 그레이 스케일 이미지
    """
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image


# 밝기 조정
def adjust_brightness(image, brightness):
    """
    이미지를 주어진 밝기값으로 조정함

    Args:
        image : 입력 이미지
        brightness : 밝기 조정 값(-100 ~ 100)

    return: 밝기가 조정된 이미지
    """
    adjusted = cv2.convertScaleAbs(image, alpha=1, beta=brightness)
    return adjusted


# 슬라이더 콜백 함수(밝기 조정)
def on_brightness_change(val):
    global img, window_name
    brightness = val - 100  # 슬라이더 값(0~200을 -100~100으로 변환)
    adjusted_img = adjust_brightness(img, brightness)
    cv2.imshow(window_name, adjusted_img)


# 대비 조정(수동 조정)
def manual_contrast_adjust(image, alpha):
    """
    입력 이미지의 대비를 조정함

    Args:
        image : 입력 이미지
        alpha : 대비 조정 계수(0.1~3.0)

    return: 대비가 조정된 이미지
    """
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    return adjusted


# 슬라이더 콜백 함수(대비 조정)
def on_contrast_change(val):
    global img, window_name
    alpha = val/100  # 슬라이더 값(10~300)을 0.1~3.0으로 변환
    adjusted_img = manual_contrast_adjust(img, alpha)
    cv2.imshow(window_name, adjusted_img)


# 영상 합성
def blend_images(image1, image2, alpha):
    """
    두 이미지를 주어진 알파 값으로 합성함

    Args:
        image1 : 첫 번째 입력 이미지
        image2 : 두 번째 입력 이미지
        alpha : 첫 번째 이미지의 비율(0.0~1.0)

    return: 합성된 이미지
    """
    beta = 1.0 - alpha  # 두 번째 이미지의 비율
    blended = cv2.addWeighted(image1, alpha, image2, beta, 0)
    return blended


# 슬라이더 콜백 함수(영상 합성)
def on_blend_change(val):
    global img1, img2, window_name
    alpha = val/100 # 슬라이더 값 (0~100)을 0.0~1.0으로 변환
    blended_img = blend_images(img1, img2, alpha)
    cv2.imshow(window_name, blended_img)


# 히스토그램 평활화
def equalize_histogram(image):
    """
    입력 이미지의 히스토그램을 평활화함

    Args:
        image : 입력 이미지

    return : 히스토그램이 평활화된 이미지
    """
    # 그레이 스케일 변환 후 히스토그램 평활화 적용
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized


# 명암 대비 조정 함수(자동 조정)
def auto_contrast_stretch(image):
    """
    입력 이미지의 명암 대비를 조정함
    manual_contrast_adjust와 달리, 히스토그램 스트레칭을 통해
    명암을 자동으로 보정해주는 기능

    Args:
        image : 입력 이미지

    return: 명암 대비가 조정된 이미지
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    min_val, max_val = np.min(gray), np.max(gray)
    print(f"Original min: {min_val}, max: {max_val}")
    
    stretched = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return stretched


# 블러링 함수
def apply_blur(image, kernel_size):
    """
    입력 이미지에 Gaussian Blur를 적용함
    

    Args:
        image : 입력 이미지
        kernel_size : 블러링 커널 크기

    return : 블러링된 이미지
    """
    if kernel_size % 2 == 0:  # 커널 크기는 홀수여야함
        raise ValueError("Kernal size must be an odd number")
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)    
    return blurred


# 슬라이더 콜백 함수(블러링)
def on_blur_change(val):
    global img, window_name
    kernel_size = val if val % 2 == 1 else val + 1
    blurred_img = apply_blur(img, kernel_size)
    cv2.imshow(window_name, blurred_img)


# 샤프닝 함수
def apply_sharpen(image):
    """
    입력 이미지에 샤프닝 효과 적용

    Args:
        image : 입력 이미지

    return : 샤프닝 된 이미지
    """
    kernel = np.array([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])
    sharpend = cv2.filter2D(image, -1, kernel)
    return sharpend


# 에지 검출
def detect_edges(image, threshold1, threshold2):
    """
    입력 이미지에서 Canny Edge Detection 수행함

    Args:
        image : 입력 이미지
        threshold1 : 에지 검출을 위한 첫 번째 임계값
        threshold2 : 에지 검출을 위한 두 번째 임계값

    return : 에지가 검출된 이미지
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return edges


# 슬라이더 콜백 함수(에지 검출)
def on_edge_change(val):
    global img, window_name, threshold1
    threshold2 = val
    edges_img = detect_edges(img, threshold1, threshold2)
    cv2.imshow(window_name, edges_img)


if __name__ == "__main__":
    input_path = "../images/input/sample.jpg"  # 입력 이미지 경로
    window_name = "Edge detection"
    
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"File is not found: {input_path}")
        print("Image loaded successfully.")
        
        threshold1 = 100
        threshold2 = 200
        
        cv2.namedWindow(window_name)
        cv2.createTrackbar("Threshold2", window_name, threshold2, 300, 
                           on_edge_change)
        on_edge_change(threshold2)
        
        sharpened_img = apply_sharpen(img)
        cv2.imshow(window_name, sharpened_img)
    
        output_path = "../images/output/sharpened.jpg"
        cv2.imwrite(output_path, sharpened_img)
        print(f"Sharpened image saved to {output_path}")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"Error: {e}")