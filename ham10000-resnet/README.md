# HAM10000 ResNet

Dự án mẫu cho việc huấn luyện mô hình ResNet trên bộ dữ liệu HAM10000.

## Cấu trúc

- data/
  - HAM10000_images/
  - HAM10000_metadata.csv
- src/
  - dataset.py
  - model.py
  - train.py
  - evaluate.py
  - utils.py
- checkpoints/
  - best_model.pth
- outputs/
  - confusion_matrix.png
  - training_curve.png

## Ghi chú

- Đây là file khởi tạo dự án.
- Thêm dữ liệu thực tế và triển khai mô hình khi cần.
