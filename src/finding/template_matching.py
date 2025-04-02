import os
import cv2
import numpy as np
from src.utils.visualization import draw_bounding_box

def get_template(folder_path: str):
    """
    Đọc tất cả các tệp ảnh từ một thư mục và trả về chúng dưới dạng danh sách.
    
    Hàm này duyệt qua tất cả các tệp trong thư mục đã chỉ định, đọc mỗi tệp
    như một ảnh bằng OpenCV, và thêm vào danh sách templates để trả về.
    
    Args:
        folder_path (str): Đường dẫn đến thư mục chứa các tệp ảnh template cần đọc.
        
    Returns:
        list: Danh sách các đối tượng ảnh (numpy.ndarray) đọc được từ các tệp trong 
              thư mục. Nếu một tệp không đọc được sẽ có giá trị None trong danh sách.
              
    Raises:
        FileNotFoundError: Nếu thư mục không tồn tại.
        
    Note:
        Hàm giả định mọi tệp trong thư mục đều là ảnh. Nếu có tệp không phải ảnh
        hoặc không được hỗ trợ bởi OpenCV, kết quả có thể chứa các giá trị None.
    """
    templates = []
    for file in os.listdir(folder_path):
        template = cv2.imread(os.path.join(folder_path, file))
        templates.append(template)
    return templates


def linear_multiscale_template_matching(image, templates, scale_range=(0.5, 1.0), scale_steps=10, 
                                       threshold_value=200, match_method=cv2.TM_CCOEFF_NORMED):
    """
    Thực hiện template matching với nhiều tỉ lệ khác nhau để tìm đối tượng trong ảnh.
    
    Hàm này tìm kiếm các đối tượng trong ảnh dựa trên một tập các template, bằng cách 
    thử nhiều tỉ lệ khác nhau cho mỗi template. Mặt nạ nhị phân được tạo ra để loại bỏ 
    nền của các template, cải thiện độ chính xác khi so khớp.
    
    Args:
        image (numpy.ndarray): Ảnh đích để tìm kiếm các đối tượng.
        templates (list): Danh sách các ảnh template dùng để tìm kiếm.
        scale_range (tuple, optional): Khoảng tỉ lệ (min, max) để thay đổi kích thước template.
                                       Mặc định là (0.5, 1.0) - từ 50% đến 100% kích thước gốc.
        scale_steps (int, optional): Số bước tỉ lệ sẽ được thử. Mặc định là 10.
        threshold_value (int, optional): Giá trị ngưỡng để tạo mặt nạ. Mặc định là 200.
        match_method (int, optional): Phương pháp so khớp template của OpenCV.
                                      Mặc định là cv2.TM_CCOEFF_NORMED.
    
    Returns:
        tuple: Gồm hai phần tử:
               - result_image (numpy.ndarray): Ảnh gốc được vẽ thêm các hình chữ nhật 
                                              xung quanh các đối tượng tìm thấy.
               - match_locations (list): Danh sách các vị trí (góc trên bên trái) của 
                                        các đối tượng đã tìm thấy.
    
    Example:
        templates = get_template("template_folder/")
        result_image, locations = linear_multiscale_template_matching(input_image, templates)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.show()
    """
    result_image = image.copy()
    
    scales = np.linspace(scale_range[0], scale_range[1], scale_steps)
    
    match_locations = []
    
   
    if len(image.shape) == 3 and templates and len(templates[0].shape) == 2:
        image_for_matching = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_for_matching = image
    
    for template in templates:
        if template is None:
            continue
            
        # Chuẩn bị template cho matching
        # Nếu template là ảnh màu (3 kênh), cần tạo mặt nạ từ ảnh xám
        if len(template.shape) == 3:
            # Tạo mặt nạ nhị phân từ template
            gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            _, template_mask = cv2.threshold(gray_template, threshold_value, 255, cv2.THRESH_BINARY_INV)
            
            if len(image_for_matching.shape) == 2:
                template_for_matching = gray_template
            else:
                template_for_matching = template
        else:
            template_mask = template.copy()
            template_for_matching = template.copy()
        
        best_correlation = -1
        best_location = None
        best_dimensions = None
        best_scale = None
        
        for scale in scales:
            scaled_template = cv2.resize(template_for_matching, (0, 0), fx=scale, fy=scale)
            scaled_mask = cv2.resize(template_mask, (scaled_template.shape[1], scaled_template.shape[0]))
            
            template_width, template_height = scaled_template.shape[1], scaled_template.shape[0]
            
            try:
                correlation_map = cv2.matchTemplate(
                    image_for_matching, 
                    scaled_template, 
                    match_method, 
                    mask=scaled_mask
                )
                
                _, max_correlation, _, max_location = cv2.minMaxLoc(correlation_map)
                
                if max_correlation > best_correlation:
                    best_correlation = max_correlation
                    best_location = max_location
                    best_dimensions = (template_width, template_height)
                    best_scale = scale
            except cv2.error as e:
                print(f"Lỗi khi thực hiện template matching với scale {scale}: {e}")
                continue
        
        if best_location is not None:
            top_left = best_location
            bottom_right = (top_left[0] + best_dimensions[0], top_left[1] + best_dimensions[1])            
            cv2.rectangle(result_image, top_left, bottom_right, (0, 0, 255), 5)            
            match_locations.append(top_left)
    
    return result_image, match_locations