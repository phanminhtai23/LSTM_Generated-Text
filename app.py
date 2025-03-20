from tensorflow.keras.models import load_model
from keras import backend as K
import streamlit as st
import time
import tensorflow as tf
import numpy as np
import collections
import re
import unicodedata
import string
# import tensorflow as tf
import tensorflow as tf

def read_data(fname):
    with open(fname, "r", encoding="utf-8") as f:
        text = f.read()
        cleaned_text = clean_text(text).split()
        return cleaned_text


def build_dataset(words):
    count = collections.Counter(words).most_common()
    word2id = {}
    for word, freq in count:
        word2id[word] = len(word2id)
        id2word = dict(zip(word2id.values(), word2id.keys()))
    return word2id, id2word


def clean_text(text):
    # 1. Chuyển về chữ thường
    text = text.lower()

    # 2. Chuẩn hóa unicode (tránh lỗi dấu tiếng Việt)
    text = unicodedata.normalize("NFC", text)

    # 3. Loại bỏ dấu câu (chấm, phẩy, ngoặc...)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 4. Loại bỏ khoảng trắng thừa
    text = re.sub(r"\s+", " ", text).strip()

    return text


def encode(sent):
    words = sent.split()
    result = []
    for w in words:
        result.append(w2i[w])
    return np.array([result])


def decode(predicted):
    return i2w[np.argmax(predicted)]


def generate_text(model, seed_text, next_words=10):
    result = seed_text
    for _ in range(next_words):
        # Chuyển thành số và đưa vào mô hình
        encoded_message = encode(seed_text)
        # print("decode: ", decode(encoded_message))

        predicted = model.predict(
            encoded_message, verbose=0)  # Dự đoán từ tiếp theo
        output_word = decode(predicted)  # Lấy từ có xác suất cao nhất

        result += " " + output_word  # Thêm từ mới vào câu

        # Cắt bỏ từ đầu và cộng thêm từ mới được sinh ra
        seed_text = " ".join(seed_text.split()[1:] + [output_word])
        # print(seed_text)

    return result

# Kiểm tra input có từ nào ngoài thư viện từ không
def input_is_legit(input):
    
    input = input.strip().lower()
    
    words = input.split()
    # print("word", len(words))
    for w in words:
        if w not in w2i:
            return False, w
    return True, ""


# Hàm chính
def main():
    global w2i, i2w
    
    # Đọc dữ liệu
    data = read_data('./Tieu_su_Elon_Musk.txt')

    w2i, i2w = build_dataset(data)
    
   # Nếu model đã load trước đó, giải phóng bộ nhớ
    if 'model' in locals() or 'model' in globals():
        del model
        K.clear_session()
        tf.keras.backend.clear_session()

    # Load lại model
    model = tf.keras.models.load_model("./lstm_model.keras")

    
    st.markdown("<h1 style='text-align: center;'>DEMO SINH TỪ BẰNG MODEL LSTM</h1>",
                unsafe_allow_html=True)
    st.write("Sinh viên: Phan Minh Tài - B2113341")
    st.markdown(
        "Dữ liệu dùng để huấn luyện: [**see here**](https://vi.wikipedia.org/wiki/Elon_Musk)")
    st.markdown(
        "Độ chính xác trên tập train: **89,4%**")

    text_input = st.text_area(
        "Nhập chuỗi từ bạn muốn model sinh ra tiếp (tiếng Việt, đề xuất là 3 từ): ", height=100, key="text_area", placeholder="ví dụ: Elon theo học")

    total_words_wants_to_generating = st.text_input(
        "Bạn muốn nhận output từ model bao nhiêu từ ? (Nếu để trống, mặc định 10 từ): ", placeholder="ví dụ: 5")
    
    convert_button = st.button("Generating")

    if convert_button:
        if text_input == '':
            st.warning(
                "Vui lòng nhập chuỗi bạn muốn model sinh từ tiếp :) !"
            )
        # Nhập số từ muốn nhận không hợp lệ
        elif total_words_wants_to_generating.strip().isdigit() is False and total_words_wants_to_generating.strip() != "":
            st.warning(
                "Vui lòng nhập số từ muốn nhận là số nguyên dương!"
            )
        # Có nhập số từ muốn nhận hoặc không
        else:
            global number_user_input
            if total_words_wants_to_generating.strip() == "":
                number_user_input = 10
            elif total_words_wants_to_generating.strip().isdigit():
                number_user_input = int(
                    total_words_wants_to_generating.strip())
            
            # text user nhập
            len_text_input = len(text_input.strip().split())
        
            # Nhập số từ muốn nhận k hợp lệ
            if number_user_input <= len_text_input:
                st.warning(
                    "Số từ bạn muốn nhận phải lớn hơn số từ bạn đã nhập!"
                )
            else:
                next_words = number_user_input - len_text_input
                result, unlegit_word = input_is_legit(text_input)
                if result:
                    # Bắt đầu đo thời gian
                    start_time = time.time()

                    # Hiển thị thanh tiến trình
                    progress_bar = st.progress(0)
                    # Sử dụng st.spinner để hiển thị thông báo đang chạy
                    with st.spinner('Model đang sinh từ...'):
                        time.sleep(0.5)
                        # dict, feature = get_feature(uploaded_email)
                        progress_bar.progress(30)
                        time.sleep(0.5)

                        respone = generate_text(
                            model, text_input.strip().lower(), next_words)
                        
                        
                        progress_bar.progress(100)
                        time.sleep(0.5)
                        st.success('Model sinh từ đã xong ! ✔️')
                        
                        st.header("Kết quả:")
                        st.write(respone)
                        
                        # Kết thúc đo thời gian
                        end_time = time.time()
                        execution_time = end_time - start_time
                        print(
                            f"Totals time: {execution_time:.2f} seconds")
                        
                else:
                    st.warning(
                        f"Có từ **{unlegit_word}** không có trong túi từ, vui lòng nhập lại input !"
                    )
if __name__ == "__main__":
    main()
