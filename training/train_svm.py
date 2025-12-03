import joblib
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from training.data_loader import load_and_process_data, MODELS_DIR

def train():
    print("\n[SVM] Bắt đầu huấn luyện (Có thể lâu)...")
    X_train, y_train = load_and_process_data()
    
    # Giảm dữ liệu riêng cho SVM nếu quá lớn để tránh treo máy
    if len(X_train) > 5000:
        print("   -> Cắt giảm dữ liệu xuống 5000 mẫu cho SVM...")
        idx = np.random.choice(len(X_train), 5000, replace=False)
        X_train_svm = X_train[idx]
        y_train_svm = y_train.iloc[idx]
    else:
        X_train_svm, y_train_svm = X_train, y_train

    base_model = SVC(kernel="rbf", probability=True, random_state=42)
    
    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
    calibrated_model.fit(X_train_svm, y_train_svm)
    
    save_path = os.path.join(MODELS_DIR, "svm.pkl")
    joblib.dump(calibrated_model, save_path)
    print("[SVM] Đã lưu model.")

if __name__ == "__main__":
    train()