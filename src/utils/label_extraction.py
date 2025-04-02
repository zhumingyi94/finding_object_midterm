import xml.etree.ElementTree as ET


def extract_objects_from_xml(xml_path: str): # Thêm type hint cho rõ ràng
    """
    Phân tích cú pháp một file label XML (định dạng PASCAL VOC)
    và trích xuất tên đối tượng (nhãn) cùng với các hộp giới hạn (bounding box).

    Args:
        xml_path (str): Đường dẫn đến file label XML.

    Returns:
        tuple: Một tuple chứa:
            - str: Tên file của ảnh liên quan (hoặc None nếu không tìm thấy).
            - list: Một danh sách các dictionary, mỗi dictionary đại diện cho
                    một đối tượng với các khóa 'name' (tên) và 'bbox' (hộp giới hạn).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_filename_tag = root.find('filename')
    image_filename = image_filename_tag.text if image_filename_tag is not None else None

    objects_data = []
    for obj in root.findall('object'):
        obj_info = {}
        name_tag = obj.find('name')

        if name_tag is None or name_tag.text is None or not name_tag.text.strip():
            print(f"Cảnh báo: Tìm thấy đối tượng không có tên hợp lệ trong {xml_path}. Bỏ qua.")
            continue # Bỏ qua nếu tên rỗng hoặc thiếu

        obj_info['name'] = name_tag.text.strip() 

        bbox_tag = obj.find('bndbox')
        if bbox_tag is not None:
            xmin_tag = bbox_tag.find('xmin')
            ymin_tag = bbox_tag.find('ymin')
            xmax_tag = bbox_tag.find('xmax')
            ymax_tag = bbox_tag.find('ymax')

            if all(tag is not None and tag.text is not None for tag in [xmin_tag, ymin_tag, xmax_tag, ymax_tag]):
                
                obj_info['bbox'] = {
                    'xmin': float(xmin_tag.text),
                    'ymin': float(ymin_tag.text),
                    'xmax': float(xmax_tag.text),
                    'ymax': float(ymax_tag.text)
                }
                objects_data.append(obj_info)

    return image_filename, objects_data