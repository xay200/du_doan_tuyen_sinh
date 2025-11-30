import pandas as pd
import numpy as np
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def train():
    print("1. Đang đọc dữ liệu...")
    try:
        data = pd.read_csv("Data_diem_thi_thpt_nam_2019.csv")
    except FileNotFoundError:
        print("LỖI: Không tìm thấy file 'Data_diem_thi_thpt_nam_2019.csv'")
        return

    # Lấy mẫu 20% nếu dữ liệu quá lớn
    if len(data) > 50000:
        data = data.sample(frac=0.2, random_state=42)

    subjects = ["Toan Hoc", "Ngu Van", "Ngoai Ngu", "Vat Ly", "Hoa Hoc", "Sinh Hoc", "Lich Su", "Dia Ly", "GDCD"]
    data[subjects] = data[subjects].fillna(0)
    data["Tong_Diem"] = data[subjects].sum(axis=1)

    # Lấy ngưỡng trung vị (Median) để cân bằng 50/50
    threshold = data["Tong_Diem"].median()
    print(f"   -> Ngưỡng điểm chuẩn (Median): {threshold}")

    data["KetQua"] = (data["Tong_Diem"] > threshold).astype(int)
    
    # Xử lý điểm liệt
    condition_liet = (data[subjects] <= 1).any(axis=1)
    data.loc[condition_liet, "KetQua"] = 0

    X = data[subjects]
    y = data["KetQua"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if not os.path.exists("models"): os.makedirs("models")
    joblib.dump(scaler, "models/scaler.pkl")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42), # Độ sâu 5 để vẽ hình đẹp
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "SVM": SVC(kernel="rbf", probability=True)
    }

    print("2. Đang huấn luyện mô hình...")
    for name, model in models.items():
        if name == "SVM" and len(X_train_scaled) > 5000:
            # SVM chạy chậm nên chỉ lấy 5000 mẫu
            idx = np.random.choice(len(X_train_scaled), 5000, replace=False)
            model.fit(X_train_scaled[idx], y_train.iloc[idx])
        else:
            model.fit(X_train_scaled, y_train)
        
        # Lưu tên file ngắn gọn
        filename = "logistic.pkl" if "Logistic" in name else \
                   "tree.pkl" if "Tree" in name else \
                   "knn.pkl" if "KNN" in name else "svm.pkl"
        
        joblib.dump(model, f"models/{filename}")
        print(f"   -> Đã lưu {name}")

    print("\n✅ HUẤN LUYỆN THÀNH CÔNG! Hãy chạy app.py")

if __name__ == "__main__":
    train()