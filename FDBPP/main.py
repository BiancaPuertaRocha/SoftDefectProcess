import tkinter as tk
from tkinter import messagebox, Menu, ttk
import os
import pandas as pd
import matplotlib.pyplot as plt
from models import random_forest_raw, random_forest_smotenc, random_forest_adasyn

# Função para carregar e exibir métricas e plotar
def process_model(selected_file, model_type, balance_method):
    df = pd.read_csv(f'data/{selected_file}')
    if model_type == "random_forest" and not balance_method:
        auc_score, y_test, y_pred = random_forest_raw(df)
    elif model_type == "random_forest" and balance_method == "SMOTE":
        auc_score, y_test, y_pred = random_forest_smotenc(df)
    elif model_type == "random_forest" and balance_method == "ADASYN":
        auc_score, y_test, y_pred = random_forest_adasyn(df)
    else:
        messagebox.showerror("Erro", "Configuração de modelo não implementada.")
        return

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)

    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'AUC': auc_score}
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'orange', 'green', 'red', 'purple'])
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.title("Model Performance Metrics")
    plt.show()

    result = {
        'algorithm': model_type, 'dataset': selected_file, 'feature selection': "None",
        'data balance': balance_method if balance_method else "None", 'accuracy': accuracy,
        'precision': precision, 'recall': recall, 'auc': auc_score, 'f1-score': f1
    }
    os.makedirs("results", exist_ok=True)
    results_path = 'results/metrics.csv'
    if os.path.exists(results_path):
        pd.DataFrame([result]).to_csv(results_path, mode='a', header=False, index=False)
    else:
        pd.DataFrame([result]).to_csv(results_path, index=False)

    messagebox.showinfo("Execução Concluída", "Modelo executado e métricas salvas com sucesso.")

# Função chamada ao clicar em "Run"
def on_run(file_combo, model_var, balance_var, balance_method_var):
    selected_file = file_combo.get()
    selected_model = model_var.get()
    selected_balance = balance_method_var.get() if balance_var.get() else None
    if selected_model == "random_forest":
        process_model(selected_file, selected_model, selected_balance)
    else:
        messagebox.showerror("Erro", "Seleção de modelo inválida ou não implementada.")

# Interface Gráfica com Tkinter
def build_interface():
    root = tk.Tk()
    root.title("Interface de Modelagem de Dados")
    root.geometry("800x600")

    menu_bar = Menu(root)
    root.config(menu=menu_bar)
    options_menu = Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Opções", menu=options_menu)
    
    tab_control = ttk.Notebook(root)
    load_tab = ttk.Frame(tab_control)
    model_tab = ttk.Frame(tab_control)
    tab_control.add(load_tab, text="Carregar Projeto")
    tab_control.add(model_tab, text="Construir Modelo")
    tab_control.pack(expand=1, fill="both")
    
    label_id = tk.Label(load_tab, text="Insira o file_id:")
    label_id.pack(pady=10)
    entry_id = tk.Entry(load_tab, width=40)
    entry_id.pack(pady=5)

    label_proj = tk.Label(load_tab, text="Insira o nome do projeto:")
    label_proj.pack(pady=10)
    entry_proj_name = tk.Entry(load_tab, width=40)
    entry_proj_name.pack(pady=5)

    def load_files(file_id, proj_name):
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            import gdown
            gdown.download(url, f'data/local_{proj_name}.csv', quiet=False)
            messagebox.showinfo("Download Completo", "Arquivo baixado e carregado com sucesso.")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao baixar ou carregar o arquivo: {e}")

    def on_download():
        file_id = entry_id.get()
        proj_name = entry_proj_name.get()
        if file_id:
            load_files(file_id, proj_name)
        else:
            messagebox.showwarning("Atenção", "Por favor, insira o file_id.")

    download_button = tk.Button(load_tab, text="Baixar Arquivo", command=on_download)
    download_button.pack(pady=20)

    label_file = tk.Label(model_tab, text="Selecione um arquivo:")
    label_file.pack(pady=10)
    file_combo = ttk.Combobox(model_tab, values=os.listdir("data"))
    file_combo.pack(pady=5)

    label_model = tk.Label(model_tab, text="Selecione o tipo de modelo:")
    label_model.pack(pady=10)
    model_var = tk.StringVar(value="random_forest")
    model_options = ["random_forest", "bagging_random_forest", "bagging_decision_tree", "voting_classifier"]
    for option in model_options:
        tk.Radiobutton(model_tab, text=option.replace("_", " ").title(), variable=model_var, value=option).pack(anchor="w")

    balance_var = tk.BooleanVar()
    balance_check = tk.Checkbutton(model_tab, text="Aplicar Data Balance", variable=balance_var)
    balance_check.pack(pady=10)
    balance_method_var = tk.StringVar(value="SMOTE")
    balance_options = ["SMOTE", "ADASYN"]
    for option in balance_options:
        tk.Radiobutton(model_tab, text=option, variable=balance_method_var, value=option).pack(anchor="w")

    run_button = tk.Button(model_tab, text="Run", command=lambda: on_run(file_combo, model_var, balance_var, balance_method_var))
    run_button.pack(pady=20)

    root.mainloop()

# Executa a interface
build_interface()


# 1t1KvW5yLhGADNlAbTygowsBfWrwogwqr - dubbo