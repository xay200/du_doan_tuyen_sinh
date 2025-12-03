import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from training.data_loader import load_and_process_data, MODELS_DIR

def train():
    print("\n[Logistic] Bắt đầu huấn luyện...")
    X_train, y_train = load_and_process_data()
    
    # Base model
    base_model = LogisticRegression(max_iter=1000, n_jobs=-1)
    
    # Calibration (Kỹ thuật nâng cao cho xác suất chuẩn)
    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
    calibrated_model.fit(X_train, y_train)
    
    save_path = os.path.join(MODELS_DIR, "logistic.pkl")
    joblib.dump(calibrated_model, save_path)
    print("[Logistic] Đã lưu model.")

if __name__ == "__main__":
    train()