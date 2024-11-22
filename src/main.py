import cv2


# 이미지 불러오기
def load_image(file_path):
    image = cv2.imread(file_path)
    if image is None:
        raise FileNotFoundError(f"File not found: {file_path}")
    return image


def save_image(image, save_path):
    cv2.imwrite(save_path, image)


# 이미지 크기 조정
def resize_image(image, scale):
    if scale <= 0:
        raise ValueError("Scale must be greater then 0.")
    
    height, width = image.shape[:2]
    new_width = int(width*scale)
    new_height = int(height*scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image


def on_scale_change(val):
    global img, window_name
    scale = val / 100  # 슬라이더 값(0~200)을 0.01~2.00으로 변환
    resized = resize_image(img, scale)
    cv2.imshow(window_name, resized)
    

# 이미지 회전
def rotate_image(image, angle):
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
    if flip_code not in [0,1]:
        raise ValueError("Filp code must be 0(vertical) or 1(horizontal).")
    return cv2.flip(image, flip_code)


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        color = param[y, x]
        print(f"Mouse clicked at ({x}, {y}) with color: {color}")


# 밝기 조정
def adjust_brightness(image, brightness):
    adjusted = cv2.convertScaleAbs(image, alpha=1, beta=brightness)
    return adjusted


# 슬라이더 콜백 함수
def on_brightness_change(val):
    global img, window_name
    brightness = val - 100
    adjusted_img = adjust_brightness(img, brightness)
    cv2.imshow(window_name, adjusted_img)

# ROI 선택 함수
def select_roi(image):
    roi = cv2.selectROI("Select ROI", image, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi
    if w==0 or h==0:
        raise ValueError("ROI not selected or has zero size.")
    return image[int(y):int(y+h), int(x):int(x+w)]

if __name__ == "__main__":
    input_path = "../images/input/sample.jpg"  # 입력 이미지 경로
    window_name = "Image Resizer"
    
    try:
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"File is not found: {input_path}")
        print("Image loaded successfully.")
        
        selected_roi = select_roi(img)
        cv2.imshow("Selected ROI", selected_roi)
        
        output_path = "../images/output/selected_roi.jpg"
        cv2.imwrite(output_path, selected_roi)
        print(f"Selected ROI saved to {output_path}")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")