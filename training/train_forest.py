import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from training.data_loader import load_and_process_data, MODELS_DIR

def train():
    print("\n[Random Forest] Bắt đầu huấn luyện...")
    X_train, y_train = load_and_process_data()
    
    # Random Forest mạnh mẽ hơn Decision Tree
    base_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    
    # Calibration
    calibrated_model = CalibratedClassifierCV(base_model, method='sigmoid', cv=3)
    calibrated_model.fit(X_train, y_train)
    
    save_path = os.path.join(MODELS_DIR, "random.pkl")
    joblib.dump(calibrated_model, save_path)
    print("[Random Forest] Đã lưu model.")

if __name__ == "__main__":
    train()