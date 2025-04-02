import cv2
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

def draw_bounding_box(
    image: np.ndarray,
    points: Tuple[Tuple[int, int], Tuple[int, int]]
) -> np.ndarray:
    """
    Vẽ một hình chữ nhật cố định (màu đỏ, độ dày 2) lên ảnh.

    Args:
        image (np.ndarray): Ảnh đầu vào (định dạng NumPy array, thường là BGR).
        points (Tuple[Tuple[int, int], Tuple[int, int]]): Một tuple chứa tọa độ
            nguyên của hai điểm góc đối diện của hình chữ nhật,
            ví dụ: ((x_top_left, y_top_left), (x_bottom_right, y_bottom_right)).

    Returns:
        np.ndarray: Một bản sao của ảnh đầu vào với hình chữ nhật đã được vẽ.
    """
    image_with_bb = image.copy()
    (x1, y1), (x2, y2) = points
    cv2.rectangle(image_with_bb, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return image_with_bb

def plot_image_grid(images, titles=None, main_title=None, figsize=None, 
                    rows=None, cols=None, cmap=None, convert_bgr2rgb=True,
                    wspace=0.3, hspace=0.3, title_pad=10):
    """
    Hiển thị nhiều ảnh theo dạng lưới với các tùy chọn định dạng.
    
    Args:
        images (list): Danh sách các ảnh cần hiển thị (numpy.ndarray).
        titles (list, optional): Danh sách các tiêu đề cho từng ảnh.
        main_title (str, optional): Tiêu đề chung cho toàn bộ figure.
        figsize (tuple, optional): Kích thước của figure (width, height) tính bằng inches.
        rows (int, optional): Số hàng trong lưới. Nếu None, sẽ tự động tính toán.
        cols (int, optional): Số cột trong lưới. Nếu None, sẽ tự động tính toán.
        cmap (str, optional): Bảng màu matplotlib cho ảnh đơn kênh.
        convert_bgr2rgb (bool, optional): Chuyển đổi từ BGR sang RGB (cho ảnh từ OpenCV).
        wspace (float, optional): Khoảng cách ngang giữa các ảnh.
        hspace (float, optional): Khoảng cách dọc giữa các ảnh.
        title_pad (int, optional): Khoảng cách giữa tiêu đề và ảnh.
        
    Returns:
        tuple: Một tuple chứa:
               - fig (matplotlib.figure.Figure): Đối tượng figure
               - axes (numpy.ndarray): Mảng các đối tượng axes
    """
    
    n_images = len(images)
    if rows is None and cols is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    elif rows is None:
        rows = int(np.ceil(n_images / cols))
    elif cols is None:
        cols = int(np.ceil(n_images / rows))
    
    if figsize is None:
        figsize = (cols * 3, rows * 3) 
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    if main_title:
        fig.suptitle(main_title, fontsize=16)
    
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break
            
        r, c = idx // cols, idx % cols
        ax = axes[r, c]
        
        if convert_bgr2rgb and img is not None and len(img.shape) == 3 and img.shape[2] == 3:
            img = img[:, :, ::-1]
        
        if img is not None:
            ax.imshow(img, cmap=cmap)
            
            if titles is not None and idx < len(titles):
                ax.set_title(titles[idx], pad=title_pad)
            
            ax.axis('off')
        
    for idx in range(n_images, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis('off')
        
    plt.tight_layout()
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if main_title:
        plt.subplots_adjust(top=0.9)
    
    return fig, axes


