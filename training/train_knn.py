import joblib
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from training.data_loader import load_and_process_data, MODELS_DIR

def train():
    print("\n[KNN] Bắt đầu huấn luyện...")
    X_train, y_train = load_and_process_data()
    
    base_model = KNeighborsClassifier(n_neighbors=20, n_jobs=-1)
    
    # KNN trả về xác suất rời rạc, Calibration giúp làm mượt
    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
    calibrated_model.fit(X_train, y_train)
    
    save_path = os.path.join(MODELS_DIR, "knn.pkl")
    joblib.dump(calibrated_model, save_path)
    print("[KNN] Đã lưu model.")

if __name__ == "__main__":
    train()