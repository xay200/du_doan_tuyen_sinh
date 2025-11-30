from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pandas as pd
import io
import base64

# --- CẤU HÌNH MATPLOTLIB KHÔNG CẦN GUI ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

app = Flask(__name__)

# --- LOAD MODELS ---
models = {}
scaler = None
COLS_ORDER = ["Toan Hoc", "Ngu Van", "Ngoai Ngu", "Vat Ly", "Hoa Hoc", "Sinh Hoc", "Lich Su", "Dia Ly", "GDCD"]

def load_resources():
    global scaler, models
    if os.path.exists("models/scaler.pkl"):
        scaler = joblib.load("models/scaler.pkl")
    
    # Mapping tên file
    file_map = {
        "Logistic Regression": "models/logistic.pkl",
        "Decision Tree": "models/tree.pkl",
        "KNN": "models/knn.pkl",
        "SVM": "models/svm.pkl"
    }
    for name, path in file_map.items():
        if os.path.exists(path):
            try: models[name] = joblib.load(path)
            except: pass

load_resources()

PRIORITY_SCORES = {"KV1": 0.75, "KV2-NT": 0.5, "KV2": 0.25, "KV3": 0.0}

def get_safe_float(val):
    try: return float(val) if val and val.strip() != "" else 0.0
    except ValueError: return 0.0

def plot_to_base64(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig) # Giải phóng bộ nhớ
    return base64.b64encode(img.getvalue()).decode()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Tải lại model nếu chưa có (phòng trường hợp chạy app trước khi train xong)
    if not models: load_resources()
    
    if not scaler or not models:
        return "<h3>LỖI: Chưa tìm thấy model. Vui lòng chạy 'python train_models.py' trước.</h3>"

    try:
        # 1. Lấy dữ liệu
        name = request.form.get("name", "Thí sinh")
        region = request.form.get("region", "KV3")
        block_name = request.form.get("block_name", "A00")
        
        scores = {
            "math": get_safe_float(request.form.get("math")),
            "literature": get_safe_float(request.form.get("literature")),
            "english": get_safe_float(request.form.get("english")),
            "physics": get_safe_float(request.form.get("physics")),
            "chemistry": get_safe_float(request.form.get("chemistry")),
            "biology": get_safe_float(request.form.get("biology")),
            "history": get_safe_float(request.form.get("history")),
            "geography": get_safe_float(request.form.get("geography")),
            "gdcd": get_safe_float(request.form.get("gdcd")),
        }

        # 2. DataFrame
        feat_map = {
            "Toan Hoc": scores["math"], "Ngu Van": scores["literature"], "Ngoai Ngu": scores["english"],
            "Vat Ly": scores["physics"], "Hoa Hoc": scores["chemistry"], "Sinh Hoc": scores["biology"],
            "Lich Su": scores["history"], "Dia Ly": scores["geography"], "GDCD": scores["gdcd"]
        }
        input_df = pd.DataFrame([feat_map])[COLS_ORDER]

        # 3. Tính điểm
        active_scores = [v for v in scores.values() if v > 0]
        is_liet = any(s <= 1.0 for s in active_scores) if active_scores else False
        
        raw_total = 0
        if block_name == "A00": raw_total = scores["math"] + scores["physics"] + scores["chemistry"]
        elif block_name == "A01": raw_total = scores["math"] + scores["physics"] + scores["english"]
        elif block_name == "B00": raw_total = scores["math"] + scores["chemistry"] + scores["biology"]
        elif block_name == "C00": raw_total = scores["literature"] + scores["history"] + scores["geography"]
        elif block_name == "D01": raw_total = scores["math"] + scores["literature"] + scores["english"]
        else: raw_total = scores["math"] + scores["literature"] + scores["english"]

        final_total = raw_total + PRIORITY_SCORES.get(region, 0)

        # 4. Dự đoán
        results = {}
        probs = []
        plot_compare = None
        plot_tree_img = None
        message = ""
        result_class = ""

        # Logic điểm liệt
        if is_liet:
            message = "TRƯỢT (ĐIỂM LIỆT)"
            result_class = "danger"
            results = {k: "Trượt (0.0%)" for k in models}
        else:
            input_scaled = scaler.transform(input_df)
            
            # Tính điểm thưởng Soft-Boost
            bonus = 0
            if final_total >= 24: bonus = 30
            elif final_total >= 21: bonus = 15

            for m_name, model in models.items():
                # Lấy xác suất gốc
                raw = model.predict_proba(input_scaled)[0][1] * 100
                
                # Cộng điểm thưởng nhưng không quá 99.9%
                final_prob = min(raw + bonus, 99.9)
                
                # Cứu điểm cao
                if final_total >= 24 and final_prob < 50: final_prob = 60
                
                label = "Đỗ" if final_prob >= 50 else "Trượt"
                results[m_name] = f"{label} ({final_prob:.1f}%)"
                probs.append(final_prob)

            # Kết luận chung
            avg_prob = np.mean(probs)
            if avg_prob >= 80:
                message = f"RẤT KHẢ QUAN ({avg_prob:.1f}%)"
                result_class = "success"
            elif avg_prob >= 50:
                message = f"CÓ CƠ HỘI ({avg_prob:.1f}%)"
                result_class = "warning"
            else:
                message = f"NGUY CƠ CAO ({avg_prob:.1f}%)"
                result_class = "danger"

            # --- VẼ BIỂU ĐỒ (Bọc trong Try-Except để không crash) ---
            try:
                # 1. Biểu đồ cột
                fig1 = plt.figure(figsize=(8, 4))
                colors = ['#4e73df', '#1cc88a', '#36b9cc', '#f6c23e']
                bars = plt.bar(list(results.keys()), probs, color=colors[:len(results)])
                plt.axhline(50, color='r', linestyle='--')
                plt.ylim(0, 110)
                plt.title("Xác suất dự đoán")
                for bar in bars:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1, f"{bar.get_height():.1f}%", ha='center')
                plot_compare = plot_to_base64(fig1)

                # 2. Biểu đồ cây (Nếu có model Tree)
                if "Decision Tree" in models:
                    fig2 = plt.figure(figsize=(20, 10))
                    plot_tree(models["Decision Tree"], feature_names=COLS_ORDER, 
                              class_names=["Trượt", "Đỗ"], filled=True, rounded=True, 
                              fontsize=10, max_depth=3)
                    plt.title("Mô hình Cây Quyết Định (Decision Tree)")
                    plot_tree_img = plot_to_base64(fig2)
            
            except Exception as e:
                print(f"Lỗi vẽ biểu đồ: {e}") 
                # Không return lỗi, vẫn tiếp tục để hiện kết quả text

        return render_template("result.html",
                               name=name, block_name=block_name,
                               final_total=round(final_total, 2),
                               message=message, result_class=result_class,
                               results=results,
                               plot_compare=plot_compare,
                               plot_tree_img=plot_tree_img,
                               is_liet=is_liet)

    except Exception as e:
        import traceback
        return f"<h3>Lỗi hệ thống:</h3><pre>{traceback.format_exc()}</pre>"

if __name__ == "__main__":
    app.run(debug=True)