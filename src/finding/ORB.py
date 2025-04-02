import cv2
import numpy as np

def orb_detect_and_visualize_matches(
    image,
    templates,
    n_features=1000,
    min_good_match_count=10,
    ratio_test_thresh=0.75,
    draw_match_flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
):
    """
    Phát hiện đặc trưng ORB, đối sánh và trực quan hóa các cặp khớp.

    Hàm này tìm và đối sánh đặc trưng ORB giữa từng template và ảnh đích.
    Thay vì vẽ bounding box, nó tạo ra các ảnh trực quan hóa hiển thị
    keypoints và các đường nối giữa các cặp đặc trưng khớp (good matches).

    Args:
        image (np.ndarray): Ảnh đích BGR.
        templates (list): Danh sách các ảnh template BGR.
        n_features (int, optional): Số đặc trưng ORB tối đa. Mặc định 1000.
        min_good_match_count (int, optional): Số lượng cặp khớp "tốt" tối thiểu
                                           (sau ratio test) để tạo ảnh trực quan.
                                           Mặc định 10.
        ratio_test_thresh (float, optional): Ngưỡng ratio test. Mặc định 0.75.
        draw_match_flags (int, optional): Cờ cho cv2.drawMatches. Mặc định là
                                          không vẽ các keypoint đơn lẻ không khớp.

    Returns:
        tuple: Gồm hai phần tử:
               - match_visualization_images (list): Danh sách các ảnh (np.ndarray)
                 hiển thị các cặp khớp cho mỗi template đạt ngưỡng khớp.
               - match_summary (list): Danh sách các dict chứa thông tin tóm tắt
                 về các kết quả khớp, gồm: 'template_idx', 'num_keypoints_template',
                 'num_keypoints_image', 'num_good_matches'.
    """
    match_visualization_images = []
    match_summary = []

    if len(image.shape) != 3:
        print("Cảnh báo: Ảnh đầu vào không phải BGR. Vẽ có thể không như ý.")
        image_color_for_drawing = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image_gray = image
    else:
        image_color_for_drawing = image.copy()
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    try:
        orb = cv2.ORB_create(nfeatures=n_features)
    except Exception as e:
         raise ImportError(f"Không thể tạo ORB: {e}")

    kp_image, des_image = orb.detectAndCompute(image_gray, None)
    if des_image is None:
        print("Không tìm thấy đặc trưng ORB nào trong ảnh đích.")
        return match_visualization_images, match_summary
    num_kp_image = len(kp_image)
    print(f"Tìm thấy {num_kp_image} keypoints trong ảnh đích.")
    if num_kp_image < min_good_match_count:
         print("Số keypoints ảnh đích quá ít.")

    print(f"Bắt đầu xử lý {len(templates)} templates...")
    for template_idx, template in enumerate(templates):
        if template is None: continue
        if len(template.shape) != 3:
            print(f"Cảnh báo: Template {template_idx} không phải BGR. Chuyển đổi tạm.")
            template_color_for_drawing = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
            template_gray = template
        else:
            template_color_for_drawing = template.copy()
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        kp_template, des_template = orb.detectAndCompute(template_gray, None)

        if des_template is None:
            print(f"Template {template_idx}: Không tìm thấy đặc trưng ORB.")
            continue
        num_kp_template = len(kp_template)
        if num_kp_template < min_good_match_count:
             print(f"Template {template_idx}: Số keypoints ({num_kp_template}) quá ít.")
             continue

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        if des_image is None:
            print("Không có descriptors trong ảnh đích để khớp.")
            continue

        if des_template.dtype != np.uint8: des_template = des_template.astype(np.uint8)
        if des_image.dtype != np.uint8: des_image = des_image.astype(np.uint8)

        raw_matches = matcher.knnMatch(des_template, des_image, k=2)

        good_matches = []
        if raw_matches:
            for match_pair in raw_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_test_thresh * n.distance:
                        good_matches.append(m)

        num_good = len(good_matches)
        print(f"Template {template_idx}: Keypoints={num_kp_template}, Good Matches={num_good}")

        if num_good >= min_good_match_count:
            print(f" -> Đạt ngưỡng, tạo ảnh trực quan hóa cho template {template_idx}...")
            img_matches = cv2.drawMatches(
                template_color_for_drawing, kp_template,
                image_color_for_drawing, kp_image,
                good_matches,
                None,
                flags=draw_match_flags
            )
            match_visualization_images.append(img_matches)
            match_summary.append({
                'template_idx': template_idx,
                'num_keypoints_template': num_kp_template,
                'num_keypoints_image': num_kp_image, # Tổng số kp ảnh đích
                'num_good_matches': num_good
            })


    return match_visualization_images, match_summary