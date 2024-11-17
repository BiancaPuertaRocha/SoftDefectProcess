import tkinter as tk
from tkinter import messagebox, Menu, ttk
import os
import pandas as pd
import matplotlib.pyplot as plt
from models import *

def process_model(selected_file, model_type, balance_method, feature_selection_method, balance_first):
    df = pd.read_csv(f'data/{selected_file}')
    
    # Inicializar variáveis de pré-processamento
    sampler = None
    feature_selector = None

    # Configuração do sampler (método de balanceamento)
    if balance_method == "SMOTE":
        sampler = balance_data_with_smotenc
    elif balance_method == "ADASYN":
        sampler = balance_data_with_adasyn

    # Configuração do feature_selector (método de seleção de características)
    if feature_selection_method == "CFS":
        feature_selector = cfs_feature_selection
    elif feature_selection_method == "FS":
        feature_selector = fisher_score_feature_selection
    elif feature_selection_method == "CST":
        feature_selector = chi_square_feature_selection

    # Determinar ordem de execução e chamar o pipeline
    if sampler is None and feature_selector is None:
        # Nenhum pré-processamento
        auc_score, y_test, y_pred = random_forest_pipeline(df)
        preprocessing_order = "None"
    elif balance_first:
        # Balanceamento seguido de seleção de atributos
        auc_score, y_test, y_pred = random_forest_pipeline(df, sampler=sampler, feature_selector=feature_selector, balance_first=balance_first)
        preprocessing_order = "Balanceamento -> Seleção de Features"
    else:
        # Seleção de atributos seguida de balanceamento
        auc_score, y_test, y_pred = random_forest_pipeline(df, sampler=sampler, feature_selector=feature_selector, balance_first=balance_first)
        preprocessing_order = "Seleção de Features -> Balanceamento"

    # Métricas de avaliação
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)

    # Visualizar métricas
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'AUC': auc_score}
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values(), color=['blue', 'orange', 'green', 'red', 'purple'])
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.title("Model Performance Metrics")
    plt.show()

    # Salvar métricas, incluindo informações de pré-processamento
    result = {
        'algorithm': model_type,
        'dataset': selected_file,
        'feature_selection': feature_selection_method if feature_selection_method != "None" else "None",
        'data_balance': balance_method if balance_method != "None" else "None",
        'preprocessing_order': preprocessing_order,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc_score,
        'f1-score': f1
    }
    os.makedirs("results", exist_ok=True)
    results_path = 'results/metrics.csv'
    if os.path.exists(results_path):
        pd.DataFrame([result]).to_csv(results_path, mode='a', header=False, index=False)
    else:
        pd.DataFrame([result]).to_csv(results_path, index=False)

    messagebox.showinfo("Execução Concluída", "Modelo executado e métricas salvas com sucesso.")

# Função para o botão "Run"
def on_run(file_combo, model_var, balance_method_var, feature_selection_var, balance_first_var):
    selected_file = file_combo.get()
    selected_model = model_var.get()
    selected_balance = balance_method_var.get() if balance_method_var.get() != "None" else None
    selected_feature_selection = feature_selection_var.get() if feature_selection_var.get() != "None" else "None"
    balance_first = balance_first_var.get()  # Define a ordem de execução
    process_model(selected_file, selected_model, selected_balance, selected_feature_selection, balance_first)

# Interface Gráfica com a aba "Carregar Projeto" restaurada
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
    
    # Aba "Carregar Projeto"
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

    # Aba "Construir Modelo"
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

    balance_method_var = tk.StringVar(value="None")
    balance_options = ["None", "SMOTE", "ADASYN"]
    balance_method_label = tk.Label(model_tab, text="Método de Balanceamento de Dados:")
    balance_method_label.pack(pady=10)
    for option in balance_options:
        tk.Radiobutton(model_tab, text=option, variable=balance_method_var, value=option).pack(anchor="w")

    # Seleção de Features (opcional)
    feature_selection_var = tk.StringVar(value="None")
    feature_options = ["None", "CST", "FS", "CFS"]
    feature_selection_label = tk.Label(model_tab, text="Seleção de Características:")
    feature_selection_label.pack(pady=10)
    for option in feature_options:
        tk.Radiobutton(model_tab, text=option, variable=feature_selection_var, value=option).pack(anchor="w")

    # Escolher ordem (feature selection primeiro ou balanceamento primeiro)
    balance_first_var = tk.BooleanVar()
    order_label = tk.Label(model_tab, text="Escolha a ordem de processamento:")
    order_label.pack(pady=10)
    tk.Radiobutton(model_tab, text="Seleção de Características primeiro", variable=balance_first_var, value=False).pack(anchor="w")
    tk.Radiobutton(model_tab, text="Balanceamento de Dados primeiro", variable=balance_first_var, value=True).pack(anchor="w")

    run_button = tk.Button(model_tab, text="Run", command=lambda: on_run(file_combo, model_var, balance_method_var, feature_selection_var, balance_first_var))
    run_button.pack(pady=20)

    root.mainloop()

# Executa a interface
build_interface()
