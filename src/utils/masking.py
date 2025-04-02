import cv2
import matplotlib.pyplot as plt

def binary_mask(template, threshold_value=200, max_value=255, threshold_type=cv2.THRESH_BINARY_INV):
    """
    Tạo mặt nạ nhị phân từ ảnh template để sử dụng trong template matching.
    
    Hàm này chuyển đổi ảnh template sang ảnh xám và áp dụng phép ngưỡng (thresholding)
    để tạo ra một mặt nạ nhị phân, giúp phân biệt đối tượng với nền khi thực hiện
    template matching.
    
    Args:
        template (numpy.ndarray): Ảnh template gốc (BGR).
        threshold_value (int, optional): Giá trị ngưỡng để phân tách đối tượng và nền.
                                        Các pixel có giá trị lớn hơn hoặc bằng threshold_value 
                                        sẽ được gán giá trị 0 khi sử dụng THRESH_BINARY_INV.
                                        Mặc định là 200.
        max_value (int, optional): Giá trị được gán cho các pixel vượt qua ngưỡng
                                  (hoặc không vượt qua khi dùng THRESH_BINARY_INV).
                                  Mặc định là 255.
        threshold_type (int, optional): Loại thuật toán ngưỡng áp dụng.
                                       Mặc định là cv2.THRESH_BINARY_INV, trong đó:
                                       - Các pixel < threshold_value sẽ được gán max_value
                                       - Các pixel >= threshold_value sẽ được gán 0
    
    Return:
        numpy.ndarray: Mặt nạ nhị phân với các giá trị 0 (đen) và max_value (trắng),
                      trong đó vùng đối tượng thường là trắng (255) và nền là đen (0)
                      khi sử dụng THRESH_BINARY_INV.
    """
    if len(template.shape) == 3 and template.shape[2] == 3:
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        gray_template = template.copy()

    _, mask = cv2.threshold(gray_template, threshold_value, max_value, threshold_type)
    
    return mask

def convert_templates_to_binary(templates, threshold_value=200, max_value=255, threshold_type=cv2.THRESH_BINARY_INV):
    """
    Chuyển đổi danh sách các template thành dạng mặt nạ nhị phân.
    
    Hàm này áp dụng hàm binary_mask cho mỗi template trong danh sách và trả về
    danh sách các mặt nạ nhị phân tương ứng.
    
    Args:
        templates (list): Danh sách các template (ảnh) cần chuyển đổi.
        threshold_value (int, optional): Giá trị ngưỡng cho hàm binary_mask.
                                        Mặc định là 200.
        max_value (int, optional): Giá trị tối đa cho hàm binary_mask.
                                  Mặc định là 255.
        threshold_type (int, optional): Loại ngưỡng cho hàm binary_mask.
                                       Mặc định là cv2.THRESH_BINARY_INV.
    
    Returns:
        list: Danh sách các mặt nạ nhị phân tương ứng với các template đầu vào.
              Nếu một template là None, mặt nạ tương ứng cũng sẽ là None.
    """
    binary_templates = []
    
    for template in templates:
        if template is None:
            binary_templates.append(None)
            continue
            
        binary_template = binary_mask(
            template,
            threshold_value=threshold_value,
            max_value=max_value,
            threshold_type=threshold_type
        )
        
        binary_templates.append(binary_template)
    
    return binary_templates


def visualize_binary_templates(original_templates, binary_templates, max_display=5):
    """
    Hiển thị các template gốc và các mặt nạ nhị phân tương ứng.
    
    Args:
        original_templates (list): Danh sách các template gốc.
        binary_templates (list): Danh sách các mặt nạ nhị phân.
        max_display (int, optional): Số lượng template tối đa để hiển thị.
                                    Mặc định là 5.
    """
    display_count = min(len(original_templates), max_display)
    
    plt.figure(figsize=(15, 4 * display_count))
    
    for i in range(display_count):
        if original_templates[i] is None or binary_templates[i] is None:
            continue
            
        # Hiển thị template gốc
        plt.subplot(display_count, 2, 2*i + 1)
        if len(original_templates[i].shape) == 3:
            plt.imshow(cv2.cvtColor(original_templates[i], cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(original_templates[i], cmap='gray')
        plt.title(f'Template gốc #{i+1}')
        plt.axis('off')
        
        # Hiển thị mặt nạ nhị phân
        plt.subplot(display_count, 2, 2*i + 2)
        plt.imshow(binary_templates[i], cmap='gray')
        plt.title(f'Binary Mask #{i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
