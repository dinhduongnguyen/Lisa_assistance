# Lisa_assistance
Thiết kế thiết bị thông minh điều khiển đèn từ xa bằng giọng nói.
chú ý: trong dự án này tất cả các file định dạng .ipynb đều được chạy trên môi trường google Colab.
# Chuẩn bị dữ liệu
Hai bộ cơ sở dữ liệu gồm bộ cơ sở dữ liệu tập lệnh điều khiển đèn và bộ cơ sở dữ liệu từ khóa đánh thức thiết bị được được trích xuất thành các đặc trưng mfcc. Hai bộ cơ sở dữ liệu trên có thể tìm thấy tại:
https://drive.google.com/drive/folders/1Ggli8oaiaKYoiBqvMNJEoRmwhnN4_M6p
Nếu bạn muốn đào tạo lại mô hình theo bộ cơ sở dữ liệu riêng của bạn hãy chạy file audio_augmentation.ipynb nếu tăng cường cơ sở dữ liệu. sau đó chạy file extract_feature.ipynb để trích các đặc trưng MFCC và lưu các đặc trưng vào file .npz
# xây dựng và đào tạo mô hình
Sau khi có được file gồm các đặc trưng của cơ sở dữ liệu, tiến hành chạy file train_wakeword.py để xây dựng, đào tạo và lưu mô hình đánh thức còn file train_MHAtt_RNN.ipynb để để xây dựng, đào tạo và lưu các mô hình nhận dạng câu lệnh điều khiển đèn.
Kết quả khi chạy 2 file trên là hai mô hình wakeword_final.tflite và mha_rnn_final.tflite
# triển khai trên thiết bị
để triển khai các mô hình nhận dạng câu lệnh điều khiển đã đào tạo trên Raspberry Pi trước tiên cần cài đặt môi trường python trên Raspberry Pi, sau đó cài đặt các thư viện python cần thiết trong file requiment.txt. Cuối cùng hãy chạy file lisa_assistance.py.