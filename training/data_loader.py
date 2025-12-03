# training/data_loader.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "Data_diem_thi_thpt_nam_2019.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def load_and_process_data():
    print("[Data Loader] Đang xử lý dữ liệu chuẩn hóa...")
    
    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        raise Exception(f"Không tìm thấy file csv tại: {DATA_PATH}")

    # Lấy mẫu đủ lớn
    if len(data) > 50000:
        data = data.sample(frac=0.5, random_state=42)

    subjects = ["Toan Hoc", "Ngu Van", "Ngoai Ngu", "Vat Ly", "Hoa Hoc", "Sinh Hoc", "Lich Su", "Dia Ly", "GDCD"]
    data[subjects] = data[subjects].fillna(0)

    # --- 1. TÍNH PHỔ ĐIỂM KHỐI (ĐỂ TRA CỨU) ---
    print("Đang cập nhật phổ điểm sạch...")
    block_definitions = {
        "A00": ["Toan Hoc", "Vat Ly", "Hoa Hoc"],
        "A01": ["Toan Hoc", "Vat Ly", "Ngoai Ngu"],
        "B00": ["Toan Hoc", "Hoa Hoc", "Sinh Hoc"],
        "C00": ["Ngu Van", "Lich Su", "Dia Ly"],
        "D01": ["Toan Hoc", "Ngu Van", "Ngoai Ngu"]
    }
    
    block_dists = {}
    for block, cols in block_definitions.items():
        # Chỉ lấy học sinh có thi khối này (điểm > 0)
        df_valid = data[cols][(data[cols] > 0).all(axis=1)]
        sums = df_valid.sum(axis=1).values
        block_dists[block] = sums
        print(f"   -> Khối {block}: Mean = {np.mean(sums):.2f}, Max = {np.max(sums)}")

    joblib.dump(block_dists, os.path.join(MODELS_DIR, "block_dists.pkl"))

    # --- 2. GÁN NHÃN TRAINING (QUAN TRỌNG) ---
    # Logic mới: Tính Max điểm của các tổ hợp khối.
    # Nếu thí sinh có bất kỳ tổ hợp nào > 18 điểm -> Coi là ĐỖ (1)
    # Điều này giúp model học được "Điểm cao = Tốt" bất kể thi khối nào
    
    # Tính điểm các khối cho mỗi thí sinh
    data["Sum_A00"] = data[block_definitions["A00"]].sum(axis=1)
    data["Sum_A01"] = data[block_definitions["A01"]].sum(axis=1)
    data["Sum_B00"] = data[block_definitions["B00"]].sum(axis=1)
    data["Sum_C00"] = data[block_definitions["C00"]].sum(axis=1)
    data["Sum_D01"] = data[block_definitions["D01"]].sum(axis=1)
    
    # Lấy điểm cao nhất trong các khối mà thí sinh đó thi
    data["Max_Score"] = data[["Sum_A00", "Sum_A01", "Sum_B00", "Sum_C00", "Sum_D01"]].max(axis=1)
    
    # Ngưỡng đỗ: 18 điểm (Mức trung bình khá của Đại học)
    # Đây là cách dạy cho AI biết: "Cứ trên 18 điểm là thí sinh tiềm năng"
    THRESHOLD_SCORE = 18.0 
    data["KetQua"] = (data["Max_Score"] >= THRESHOLD_SCORE).astype(int)
    
    # Check liệt (<=1)
    condition_liet = (data[subjects] <= 1).any(axis=1)
    data.loc[condition_liet, "KetQua"] = 0

    print(f"   -> Tỉ lệ Đỗ (>= {THRESHOLD_SCORE}đ)/Trượt trong tập train: {data['KetQua'].mean():.2%}")

    # 3. Split & Scale
    X = data[subjects]
    y = data["KetQua"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    
    return X_train_scaled, y_train