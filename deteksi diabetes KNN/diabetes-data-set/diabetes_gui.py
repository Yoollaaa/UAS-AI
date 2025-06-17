import numpy as np
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import tkinter as tk
from tkinter import messagebox

#Load dan Preprocessing Data 
def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['Outcome']).values
    y = df['Outcome'].values
    return X, y

def preprocess(X_train, X_test):
    scaler = preprocessing.StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    return X_train_norm, X_test_norm, scaler

# Bagian 2: Evaluasi dan Pencarian K 
def evaluate_kknn(X_train, y_train, X_test, y_test, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    acc_test = metrics.accuracy_score(y_test, y_pred_test)
    acc_train = metrics.accuracy_score(y_train, y_pred_train)
    return acc_train, acc_test, model

def search_best_k(X_train, y_train, X_test, y_test, max_k=30):
    best_k = 1
    best_acc = 0
    for k in range(1, max_k + 1):
        _, acc_test, _ = evaluate_kknn(X_train, y_train, X_test, y_test, k)
        if acc_test > best_acc:
            best_acc = acc_test
            best_k = k
    print(f"Best k = {best_k}, Test Accuracy = {best_acc:.4f}")
    return best_k

# Bagian 3: Fungsi GUI 
def run_gui(model, scaler):
    def predict_diabetes():
        try:
            values = [float(entries[field].get()) for field in fields]
            sample = np.array(values).reshape(1, -1)
            sample_scaled = scaler.transform(sample)
            result = model.predict(sample_scaled)[0]
            hasil = "POSITIF Diabetes" if result == 1 else "NEGATIF Diabetes"
            messagebox.showinfo("Hasil Prediksi", hasil)
        except Exception as e:
            messagebox.showerror("Error", f"Input tidak valid.\n{e}")

    root = tk.Tk()
    root.title("Prediksi Diabetes (KNN)")

    fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    entries = {}

    for i, field in enumerate(fields):
        tk.Label(root, text=field).grid(row=i, column=0, padx=5, pady=5, sticky='e')
        entry = tk.Entry(root)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entries[field] = entry

    tk.Button(root, text="Cek Hasil", command=predict_diabetes).grid(row=len(fields), column=0, columnspan=2, pady=10)
    root.mainloop()

# Bagian 4: Main
def main():
    # Path ke dataset
    path = r"d:\here\4\Kecerdasan Buatan\ai\Sistem-Pendeteksi-Dini-Diabetes\diabetes-data-set\diabetes.csv"

    # Load dan preprocessing data
    X, y = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_norm, X_test_norm, scaler = preprocess(X_train, X_test)

    print("Train/Test split:", X_train.shape, X_test.shape)

    # Cari k terbaik
    best_k = search_best_k(X_train_norm, y_train, X_test_norm, y_test, max_k=30)

    # Evaluasi akhir dengan k terbaik
    acc_train, acc_test, model = evaluate_kknn(X_train_norm, y_train, X_test_norm, y_test, best_k)
    print(f"Final evaluation with k={best_k}: Train Acc={acc_train:.4f}, Test Acc={acc_test:.4f}")

    # Jalankan GUI
    run_gui(model, scaler)

if __name__ == "__main__":
    main()
