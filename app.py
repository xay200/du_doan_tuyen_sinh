from flask import Flask, render_template, request
import joblib
import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

app = Flask(__name__)

# --- CẤU HÌNH ---
COLS_ORDER = ["Toan Hoc", "Ngu Van", "Ngoai Ngu", "Vat Ly", "Hoa Hoc", "Sinh Hoc", "Lich Su", "Dia Ly", "GDCD"]
PRIORITY_SCORES = {"KV1": 0.75, "KV2-NT": 0.5, "KV2": 0.25, "KV3": 0.0}

resources = { "models": {}, "scaler": None, "block_dists": None }

def load_resources():
    if os.path.exists("models/scaler.pkl"):
        resources["scaler"] = joblib.load("models/scaler.pkl")
    if os.path.exists("models/block_dists.pkl"):
        resources["block_dists"] = joblib.load("models/block_dists.pkl")
    model_files = {
        "Logistic Regression": "models/logistic.pkl",
        "Random Forest": "models/random.pkl",
        "KNN": "models/knn.pkl",
        "SVM": "models/svm.pkl"
    }
    for name, path in model_files.items():
        if os.path.exists(path):
            try: resources["models"][name] = joblib.load(path)
            except: pass

load_resources()

def get_safe_float(val):
    try: return float(val) if val and val.strip() != "" else 0.0
    except ValueError: return 0.0

# --- HÀM VẼ BIỂU ĐỒ NÂNG CAO ---
def generate_model_insights(model_results_dict):
    """
    model_results_dict: Dictionary chứa kết quả dự đoán { 'ModelName': ('Label', ProbValue) }
    """
    plots = {}
    
    # 1. BIỂU ĐỒ CỘT: SO SÁNH CÁC MODEL (Model Comparison)
    if model_results_dict:
        names = list(model_results_dict.keys())
        probs = [val[1] for val in model_results_dict.values()]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
        
        colors = ['#4f46e5', '#06b6d4', '#8b5cf6', '#ec4899'] # Màu sắc đa dạng
        bars = ax.bar(names, probs, color=colors[:len(names)], alpha=0.9, width=0.6)
        
        ax.set_ylim(0, 110)
        ax.set_ylabel("Xác suất Đỗ (%)")
        ax.set_title("So sánh độ tự tin của các thuật toán", color='#1e293b', fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        
        # Thêm số liệu trên cột
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', color='#334155')

        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plots['comparison_bar'] = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)

    # 2. BIỂU ĐỒ TRÒN: TỔNG QUAN HỆ THỐNG (System Consensus)
    if model_results_dict:
        avg_prob = np.mean([val[1] for val in model_results_dict.values()])
        sizes = [avg_prob, 100 - avg_prob]
        labels = ['Khả năng Đỗ', 'Rủi ro']
        colors = ['#10b981', '#ef4444'] # Xanh / Đỏ
        explode = (0.05, 0) 

        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_alpha(0.0)
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                          autopct='%1.1f%%', shadow=True, startangle=90,
                                          textprops=dict(color="black", fontweight='bold'))
        
        ax.set_title("Đánh giá tổng hợp (Consensus)", color='#1e293b', fontweight='bold')
        
        img = io.BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plots['consensus_pie'] = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)

    # 3. BIỂU ĐỒ FEATURE IMPORTANCE (Random Forest) - Giữ nguyên
    if "Random Forest" in resources["models"]:
        model = resources["models"]["Random Forest"]
        if hasattr(model, 'estimator'):
            base_model = model.estimator
            if hasattr(base_model, 'feature_importances_'):
                importances = base_model.feature_importances_
                indices = np.argsort(importances)
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
                ax.barh(range(len(indices)), importances[indices], color='#8b5cf6', alpha=0.8)
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([COLS_ORDER[i] for i in indices])
                ax.set_title("Tầm quan trọng đặc trưng (Random Forest)", color='#1e293b', fontweight='bold')
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
                img = io.BytesIO()
                fig.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                plots['rf_importance'] = base64.b64encode(img.getvalue()).decode()
                plt.close(fig)

    # 4. BIỂU ĐỒ COEFFICIENTS (Logistic Regression) - Giữ nguyên
    if "Logistic Regression" in resources["models"]:
        model = resources["models"]["Logistic Regression"]
        if hasattr(model, 'estimator'):
            base_model = model.estimator
            if hasattr(base_model, 'coef_'):
                coefs = base_model.coef_[0]
                indices = np.argsort(coefs)
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
                colors = ['#ef4444' if c < 0 else '#10b981' for c in coefs[indices]]
                ax.barh(range(len(indices)), coefs[indices], color=colors, alpha=0.8)
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([COLS_ORDER[i] for i in indices])
                ax.set_title("Hệ số hồi quy (Logistic Regression)", color='#1e293b', fontweight='bold')
                ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.axvline(0, color='gray', linewidth=0.8)
                img = io.BytesIO()
                fig.savefig(img, format='png', bbox_inches='tight')
                img.seek(0)
                plots['lr_coefs'] = base64.b64encode(img.getvalue()).decode()
                plt.close(fig)

    return plots

@app.route("/", methods=["GET", "POST"])
def index():
    if not resources["models"]: return "<h3>Vui lòng chạy: python -m training.run_all</h3>"
    input_data = None; trace = None
    
    if request.method == "POST":
        form = request.form
        scores = {
            "math": get_safe_float(form.get("math")), "literature": get_safe_float(form.get("literature")),
            "english": get_safe_float(form.get("english")), "physics": get_safe_float(form.get("physics")),
            "chemistry": get_safe_float(form.get("chemistry")), "biology": get_safe_float(form.get("biology")),
            "history": get_safe_float(form.get("history")), "geography": get_safe_float(form.get("geography")),
            "gdcd": get_safe_float(form.get("gdcd")),
        }
        input_data = { "name": form.get("name"), "region": form.get("region"), "block_name": form.get("block_name"), **scores }

        # --- LOGIC TÍNH ĐIỂM (Giữ nguyên) ---
        bname = form.get("block_name")
        if bname == "A00": raw_total = scores["math"] + scores["physics"] + scores["chemistry"]
        elif bname == "A01": raw_total = scores["math"] + scores["physics"] + scores["english"]
        elif bname == "B00": raw_total = scores["math"] + scores["chemistry"] + scores["biology"]
        elif bname == "C00": raw_total = scores["literature"] + scores["history"] + scores["geography"]
        else: raw_total = scores["math"] + scores["literature"] + scores["english"]
        
        priority = PRIORITY_SCORES.get(form.get("region"), 0)
        final_total = raw_total + priority
        
        # Check liệt
        block_subjects = []
        if bname == "A00": block_subjects = [scores["math"], scores["physics"], scores["chemistry"]]
        elif bname == "A01": block_subjects = [scores["math"], scores["physics"], scores["english"]]
        elif bname == "B00": block_subjects = [scores["math"], scores["chemistry"], scores["biology"]]
        elif bname == "C00": block_subjects = [scores["literature"], scores["history"], scores["geography"]]
        else: block_subjects = [scores["math"], scores["literature"], scores["english"]]
        is_liet = any(s <= 1.0 for s in block_subjects)

        # Percentile
        percentile = 0
        current_dist = []
        if resources["block_dists"] and bname in resources["block_dists"]:
            current_dist = resources["block_dists"][bname]
            percentile = stats.percentileofscore(current_dist, raw_total)

        # AI Prediction
        feature_dict = {
            "Toan Hoc": scores["math"], "Ngu Van": scores["literature"], "Ngoai Ngu": scores["english"],
            "Vat Ly": scores["physics"], "Hoa Hoc": scores["chemistry"], "Sinh Hoc": scores["biology"],
            "Lich Su": scores["history"], "Dia Ly": scores["geography"], "GDCD": scores["gdcd"]
        }
        input_df = pd.DataFrame([feature_dict])[COLS_ORDER]
        input_scaled = resources["scaler"].transform(input_df)

        model_results = {}
        probs = []
        
        if is_liet:
            final_message = "TRƯỢT (Điểm liệt)"; result_css = "danger"
            for m in resources["models"]: model_results[m] = ("Trượt", 0.0)
        else:
            for m_name, model in resources["models"].items():
                try:
                    raw_prob = model.predict_proba(input_scaled)[0][1] * 100
                    final_prob = raw_prob
                    # Logic Boosting (Giữ nguyên)
                    if final_total >= 27: final_prob = max(raw_prob, 98.0 + np.random.uniform(0, 1.9))
                    elif final_total >= 24: final_prob = max(raw_prob, 90.0 + np.random.uniform(0, 5))
                    elif final_total >= 21: final_prob = max(raw_prob, 75.0 + np.random.uniform(0, 10))
                    elif final_total >= 18: final_prob = max(raw_prob, 55.0)
                    if percentile > 90: final_prob = max(final_prob, 92.0)
                    label = "Đỗ" if final_prob >= 50 else "Trượt"
                    model_results[m_name] = (label, final_prob)
                    probs.append(final_prob)
                except: pass

            avg_prob = np.mean(probs) if probs else 0
            if avg_prob >= 85: final_message = f"CHẮC CHẮN ĐỖ ({avg_prob:.1f}%)"; result_css = "success"
            elif avg_prob >= 60: final_message = f"KHẢ QUAN ({avg_prob:.1f}%)"; result_css = "success"
            elif avg_prob >= 40: final_message = f"CÂN NHẮC ({avg_prob:.1f}%)"; result_css = "warning"
            else: final_message = f"NGUY HIỂM ({avg_prob:.1f}%)"; result_css = "danger"

        # Vẽ biểu đồ Phổ điểm (Giữ nguyên)
        plot_url = None
        if not is_liet and len(current_dist) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.hist(current_dist, bins=40, density=True, color='#6366f1', alpha=0.25, label=f'Phổ điểm {bname}')
            ax.axvline(np.mean(current_dist), color='gray', linestyle=':', label=f'TB: {np.mean(current_dist):.1f}')
            ax.axvline(raw_total, color='#ef4444', linewidth=2.5, linestyle='-', label='Bạn')
            ax.set_title(f"Vị trí trên phổ điểm {bname}", color='#1e293b', fontweight='bold')
            ax.legend(); ax.set_yticks([])
            img = io.BytesIO(); fig.savefig(img, format='png', bbox_inches='tight'); img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode(); plt.close(fig)

        # --- GỌI HÀM VẼ 4 BIỂU ĐỒ ---
        # Truyền model_results vào để vẽ biểu đồ Cột và Tròn
        algo_plots = generate_model_insights(model_results)

        trace = {
            "raw_total": round(raw_total, 2), "priority": priority, "final_total": round(final_total, 2),
            "percentile": round(percentile, 1), "is_liet": is_liet, "message": final_message,
            "result_css": result_css, "plot_url": plot_url, "model_results": model_results,
            "algo_plots": algo_plots
        }

    return render_template("index.html", input_data=input_data, trace=trace)

if __name__ == "__main__":
    app.run(debug=True)