# Báo cáo Thực hành Xử lý Ảnh: Xác định Vật thể bằng Template Matching

**Tác giả:** Đỗ Minh Nhật (22022537)
**GitHub Repository:** https://github.com/zhumingyi94/finding_object_midterm

## Giới thiệu

Repository này chứa mã nguồn và báo cáo cho bài thực hành môn Xử lý Ảnh, tập trung vào việc xác định các vật thể ẩn trong một ảnh bằng kỹ thuật **Template Matching**, bao gồm cả phương pháp **Template Matching đa tỉ lệ (Multi-scale Template Matching)**.

## Bài toán

[source: 3] Mục tiêu là xác định vị trí của 15 vật thể được giấu trong ảnh `finding_01.jpg` (xem trong thư mục `data/images/`).

## Cấu trúc Thư mục

.
├── data/                  # Thư mục chứa dữ liệu
│   ├── ground_truth/      # Chứa file XML chú thích vị trí thực (ground truth) [source: 5, 6]
│   │   └── finding_01.xml
│   ├── images/            # Chứa ảnh đầu vào [source: 7]
│   │   └── finding_01.jpg
│   └── templates/         # Chứa ảnh mẫu (template) [source: 8]
│       └── finding_01/    # Template cho ảnh finding_01 [source: 9]
│           ├── balloon.png
│           ├── bow_tie.png
│           ├── car.png
│           └── ... (các template khác)
├── notebooks/             # Chứa Jupyter Notebooks để thử nghiệm và trực quan hóa [source: 10]
│   └── finding_01.ipynb
├── src/                   # Mã nguồn Python chính
│   ├── finding/           # Các module thuật toán tìm kiếm
│   │   ├── ORB.py         # Module cho thuật toán ORB
│   │   └── template_matching.py # Module cài đặt Template Matching đa tỉ lệ
│   └── utils/             # Các hàm tiện ích
│       ├── label_extraction.py # Xử lý file ground truth
│       ├── masking.py     # Tạo và áp dụng mặt nạ (mask)
│       └── visualization.py # Vẽ kết quả (bounding box,...)
├── README.md              # File này
└── requirements.txt       # Các thư viện Python cần thiết


## Phương pháp

Dự án này chủ yếu sử dụng kỹ thuật **Template Matching** để định vị các ảnh mẫu (templates) trong ảnh nguồn.

1.  **Template Matching cơ bản**: Kỹ thuật này trượt ảnh mẫu trên ảnh nguồn và tính toán độ tương đồng tại mỗi vị trí. [source: 28] Các phương pháp đo độ tương đồng khác nhau của OpenCV (`cv2.TM_SQDIFF`, `cv2.TM_CCORR`, `cv2.TM_CCOEFF` và các phiên bản chuẩn hóa) đã được thử nghiệm. [source: 29]
2.  **Template Matching Đa Tỉ Lệ (Multi-scale Template Matching)**: Để giải quyết vấn đề về sự khác biệt kích thước giữa template và đối tượng trong ảnh, phương pháp này thay đổi kích thước template qua nhiều tỉ lệ khác nhau và thực hiện matching tại mỗi tỉ lệ. [source: 33, 34] Hàm `linear_multiscale_template_matching` trong `src/finding/template_matching.py` thực hiện việc này.
3.  **Sử dụng Mặt nạ (Masking)**: Mặt nạ được tạo từ template để chỉ tập trung so khớp vào các pixel thuộc đối tượng, bỏ qua nền, giúp cải thiện độ chính xác.

## Kết quả và Nhận xét

* Các phương pháp Template Matching **chuẩn hóa** (`NORMED`), đặc biệt là `cv2.TM_CCORR_NORMED` và `cv2.TM_CCOEFF_NORMED`, cho kết quả tốt hơn đáng kể so với các phương pháp không chuẩn hóa, vì chúng có khả năng chống chịu tốt hơn với sự thay đổi về điều kiện ánh sáng và độ tương phản.
* Phương pháp `cv2.TM_CCORR_NORMED` được đánh giá là đáng tin cậy nhờ khả năng chống chịu thay đổi độ sáng/tương phản và thang đo tương đồng rõ ràng ([0, 1] hoặc [-1, 1]), giúp dễ đặt ngưỡng phát hiện.
* Mặc dù `cv2.TM_CCOEFF_NORMED` là phương pháp mạnh nhất được thử nghiệm, kết quả vẫn còn hạn chế. [source: 61, 62] Thuật toán gặp khó khăn trong việc phát hiện một số đối tượng, ví dụ như quả bóng bay màu đỏ, có thể do màu sắc tương đồng với nền hoặc sự thay đổi về góc nhìn/kích thước không nằm trong dải tỉ lệ thử nghiệm.
* Điều này cho thấy giới hạn của Template Matching khi đối mặt với các biến thể phức tạp về hình dạng, màu sắc và bối cảnh.

## Cài đặt và Sử dụng

1.  **Clone repository:**
    ```bash
    git clone [https://github.com/zhumingyi94/finding_object_midterm.git](https://github.com/zhumingyi94/finding_object_midterm.git)
    cd finding_object_midterm
    ```
2.  **Cài đặt thư viện:**
    Đảm bảo bạn đã cài đặt Python và pip. Sau đó chạy lệnh sau để cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```
    *(Lưu ý: Các thư viện chính có thể bao gồm `opencv-python`, `numpy`, `matplotlib`)*
3.  **Chạy thử nghiệm:**
    Mở và chạy các cell trong file Jupyter Notebook `notebooks/finding_01.ipynb` để xem quy trình xử lý, thực hiện thuật toán và trực quan hóa kết quả.

## Kiến thức thu được

* Hiểu và cài đặt được thuật toán Template Matching, bao gồm cả phiên bản đa tỉ lệ.
* So sánh và đánh giá hiệu quả của các phương pháp đo độ tương đồng khác nhau trong OpenCV.
* Hiểu được ưu điểm và hạn chế của Template Matching trong bài toán phát hiện đối tượng thực tế.
