https://github.com/dinhduongnguyen/Lisa_assistance.git
# Lisa_assistance
Thiết kế thiết bị thông minh điều khiển đèn từ xa bằng giọng nói.
chú ý: trong dự án này tất cả các file định dạng .ipynb đều được chạy trên môi trường google Colab.
# Chuẩn bị dữ liệu
Hai bộ cơ sở dữ liệu gồm bộ cơ sở dữ liệu tập lệnh điều khiển đèn và bộ cơ sở dữ liệu từ khóa đánh thức thiết bị được được trích xuất thành các đặc trưng mfcc. Hai bộ cơ sở dữ liệu trên có thể tìm thấy tại:
https://drive.google.com/drive/folders/1Ggli8oaiaKYoiBqvMNJEoRmwhnN4_M6p
Nếu bạn muốn đào tạo lại mô hình theo bộ cơ sở dữ liệu riêng của bạn hãy chạy file audio_augmentation.ipynb nếu tăng cường cơ sở dữ liệu. sau đó chạy file extract_feature.ipynb để trích các đặc trưng MFCC và lưu các đặc trưng vào file .npz
# Xây dựng và đào tạo mô hình
Sau khi có được file gồm các đặc trưng của cơ sở dữ liệu, tiến hành chạy file train_wakeword.py để xây dựng, đào tạo và lưu mô hình đánh thức còn file train_MHAtt_RNN.ipynb để để xây dựng, đào tạo và lưu các mô hình nhận dạng câu lệnh điều khiển đèn.
Kết quả khi chạy 2 file trên là hai mô hình wakeword_final.tflite và mha_rnn_final.tflite
# Thử nghiệm mô hình trên máy tính
Để chạy thử nghiệm mô hình trên máy tính, đầu tiên bạn cần một IDE có cài môi trường python 3.10. 
Sau đó hãy cài các thư viện python cần thiết bằng lệnh "pip install -r requiment0.txt" trên cmd.
Sau khi cài đặt xong các thư viện, bật microphone và loa máy tính của bạn và chạy file test_model.py để thử nghiệm mô hình.

Hướng dẫn thử nghiệm:

Khi chạy file test_model.py thành công, lúc này mô hình wakeword sẽ hoạt động bạn cần nói từ khóa "lisa ơi" để chương trình chuyển sang dùng mô hình nhận dạng câu lệnh. Sau khi nói từ khóa "lisa ơi" chương trình phản hồi "sẵn sàng" tức là mô hình wakeword nhận diện thành công từ khóa đánh thức và chương trình chuyển sang mô hình nhận dạng câu lệnh.
Đối với mô hình nhận dạng câu lệnh các câu lệnh điều khiển bao gồm "đèn bật", "đèn tắt", "đèn tăng sáng", "đèn giảm sáng", "đèn chuyển màu". Với mỗi câu lệnh được nhận dạng, chương trình sẽ có phản hồi tương ứng.
Trong 30s nếu không có câu lệnh nào được mô hình nhận dạng, chương trình sẽ chuyển sang dùng mô hình wake word và sẽ có tiếng "kết thúc" để thông báo.
# Triển khai trên thiết bị
để triển khai các mô hình nhận dạng câu lệnh điều khiển đã đào tạo trên Raspberry Pi trước tiên cần cài đặt môi trường python trên Raspberry Pi, sau đó cài đặt các thư viện python cần thiết trong file requiment.txt. Cuối cùng hãy chạy file lisa_assistance.py.