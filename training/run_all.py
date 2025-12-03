import time
# Import các hàm train từ các file con
from training import train_logistic, train_forest, train_knn, train_svm

def main():
    start = time.time()
    print("BẮT ĐẦU PIPELINE HUẤN LUYỆN TOÀN DIỆN")
    print("="*40)
    
    # Chạy lần lượt
    train_logistic.train()
    train_forest.train()
    train_knn.train()
    train_svm.train()
    
    print("="*40)
    print(f"HOÀN TẤT TOÀN BỘ! Tổng thời gian: {time.time() - start:.2f} giây")

if __name__ == "__main__":
    main()