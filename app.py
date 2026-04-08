# GUI
from tkinter import messagebox, Text, END, Label, Scrollbar
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk

import os
import joblib
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from gensim.models import Word2Vec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


from sklearn.ensemble import StackingClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
import lightgbm as lgb
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import simpledialog, messagebox
import os
import joblib
import tkinter as tk
from tkinter import filedialog, Text

warnings.filterwarnings('ignore')

current_plot_registry = []
current_plot_index = 0
graph_trigger_button = None
graph_menu = None
graph_selected_label = None
graph_status_label = None
graph_viewer_window = None
graph_canvas_container = None
graph_canvas_widget = None
graph_viewer_status_label = None
current_role = None
current_theme = 'light'
login_screen_frame = None
dashboard_sections = []
role_status_label = None
theme_toggle_button = None
guide_button = None
logout_button = None
graph_prev_button = None
graph_next_button = None
active_login_window = None

THEMES = {
    'light': {
        'window_bg': '#EEF4FF',
        'header_bg': '#133E87',
        'header_fg': '#FFFFFF',
        'card_bg': '#FFFFFF',
        'panel_bg': '#DDEBFF',
        'panel_fg': '#102542',
        'muted_fg': '#385170',
        'button_bg': '#133E87',
        'button_fg': '#FFFFFF',
        'button_active_bg': '#0E2E66',
        'action_bg': '#E8F1FF',
        'action_fg': '#102542',
        'action_active_bg': '#C6DDFF',
        'output_bg': '#0B1D3A',
        'output_fg': '#E9F1FF',
    },
    'dark': {
        'window_bg': '#0E1525',
        'header_bg': '#1E2A44',
        'header_fg': '#F4F7FF',
        'card_bg': '#17233A',
        'panel_bg': '#1A2942',
        'panel_fg': '#E3ECFF',
        'muted_fg': '#A8B7D9',
        'button_bg': '#3D6ED4',
        'button_fg': '#FFFFFF',
        'button_active_bg': '#2F58AF',
        'action_bg': '#243754',
        'action_fg': '#E3ECFF',
        'action_active_bg': '#2F4668',
        'output_bg': '#061125',
        'output_fg': '#DDE7FF',
    }
}

global MODEL_DIR, filename, X, Y, model, categories


global MODEL_DIR
global filename
global X, Y
global model
global categories


def uploadDataset():
    global df
    text.delete('1.0', END)
    file_path = filedialog.askopenfilename(
        initialdir=".",
        title="Select Dataset CSV File",
        filetypes=(("CSV Files", "*.csv"),)
    )
    if file_path:
        df = pd.read_csv(file_path)
        text.insert(END, "Dataset loaded\n")
        text.insert(END, df)
        text.insert(END, f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

    
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

W2V_DIM = 100
W2V_PATH = os.path.join(MODEL_DIR, "word2vec.model")


# ---------------- Word2Vec Training ----------------
def train_word2vec_tkinter(sentences, dim=W2V_DIM):
    global text
    text.insert('end', "Training Word2Vec model...\n")
    tokenized = [str(s).split() for s in sentences]
    model = Word2Vec(
        sentences=tokenized,
        vector_size=dim,
        window=5,
        min_count=1,
        workers=4,
        sg=1
    )
    model.save(W2V_PATH)
    text.insert('end', f"Word2Vec model saved to {W2V_PATH}\n")
    return model


# ---------------- Word2Vec Loading ----------------
def load_word2vec_tkinter():
    global text
    text.insert('end', "Loading Word2Vec model...\n")
    model = Word2Vec.load(W2V_PATH)
    text.insert('end', "Word2Vec model loaded.\n")
    return model


# ---------------- Sentence to Word2Vec ----------------
def sentence_to_word2vec_tkinter(sentence, model, dim=W2V_DIM):
    words = str(sentence).split()
    vecs = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)


# ---------------- Preprocessing (Training / Prediction) ----------------
def preprocess_data_word2vec(df, is_train=True, w2v_model=None, apply_smote=True, random_state=42):
    text.delete('1.0', END)
    
    global X, y
    text.insert('end', "Starting preprocessing...\n")
    df = df.copy()

    # Drop unnecessary columns
    df.drop(columns=['scam_type'], errors='ignore', inplace=True)

    # Encode target
    target_col = 'label'
    y = None
    le_target_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    if is_train and target_col in df.columns:
        le_target = LabelEncoder()
        df[target_col] = le_target.fit_transform(df[target_col].astype(str))
        joblib.dump(le_target, le_target_path)
        y = df[target_col].values
        text.insert('end', "Target label encoded and saved.\n")
    elif not is_train and target_col in df.columns:
        le_target = joblib.load(le_target_path)
        df[target_col] = le_target.transform(df[target_col].astype(str))
        y = df[target_col].values
        text.insert('end', "Target label loaded and transformed.\n")

    # Encode categorical columns
    categorical_cols = ['platform', 'urgency_level', 'contains_link', 'btc_address_present']
    for col in categorical_cols:
        encoder_path = os.path.join(MODEL_DIR, f"{col}_encoder.pkl")
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            joblib.dump(le, encoder_path)
            text.insert('end', f"Categorical column '{col}' encoded and saved.\n")
        else:
            le = joblib.load(encoder_path)
            df[col] = le.transform(df[col].astype(str))
            text.insert('end', f"Categorical column '{col}' loaded and transformed.\n")

    # Scale numeric features
    numeric_cols = ['promised_return_pct', 'sentiment_score', 'message_length', 'hour', 'dayofweek', 'is_weekend']
    X_structured = df[categorical_cols + numeric_cols].copy()
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if is_train:
        scaler = StandardScaler()
        X_structured[numeric_cols] = scaler.fit_transform(X_structured[numeric_cols])
        joblib.dump(scaler, scaler_path)
        text.insert('end', "Numeric columns scaled and scaler saved.\n")
    else:
        scaler = joblib.load(scaler_path)
        X_structured[numeric_cols] = scaler.transform(X_structured[numeric_cols])
        text.insert('end', "Numeric columns scaled using saved scaler.\n")

    # Word2Vec
    if is_train:
        w2v_model = train_word2vec_tkinter(df['message_text'])
    elif w2v_model is None:
        w2v_model = load_word2vec_tkinter()

    X_w2v = np.vstack(df['message_text'].apply(lambda x: sentence_to_word2vec_tkinter(x, w2v_model)))

    # Combine structured + word2vec features
    X = np.hstack([X_w2v, X_structured.values])
    text.insert('end', f"Combined Word2Vec + structured features → X shape: {X.shape}\n")

    # Apply SMOTE only during training
    if is_train and apply_smote and y is not None:
        smote = SMOTE(random_state=random_state)
        X, y = smote.fit_resample(X, y)
        text.insert('end', f"SMOTE applied → Total samples: {len(y)}\n")

    text.insert('end', "Preprocessing completed.\n")
    return X, y, w2v_model


def _ensure_dataset_loaded():
    if 'df' not in globals():
        messagebox.showwarning("Dataset Missing", "Please upload dataset first.")
        return False
    return True


def plot_eda_distribution():
    if not _ensure_dataset_loaded():
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    df['label'].value_counts().sort_index().plot(kind='bar', color=['#5B8DEF', '#FF6B6B'], ax=ax)
    ax.set_xlabel('Label')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Scam vs Legit Messages')
    fig.tight_layout()
    return fig


def plot_eda_platform():
    if not _ensure_dataset_loaded():
        return None
    fig, ax2 = plt.subplots(figsize=(9, 6))
    sns.countplot(x='platform', hue='label', data=df, palette='Set2', ax=ax2)
    ax2.set_title('Platform vs Scam/Legit')
    ax2.tick_params(axis='x', rotation=30)
    for p in ax2.patches:
        height = p.get_height()
        ax2.annotate(f"{height}", (p.get_x() + p.get_width() / 2.0, height),
                     ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    return fig


def plot_eda_sentiment_vs_length():
    if not _ensure_dataset_loaded():
        return None
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        x='sentiment_score',
        y='message_length',
        hue='label',
        data=df,
        palette='Set1',
        ax=ax
    )
    ax.set_title('Sentiment Score vs Message Length')
    fig.tight_layout()
    return fig


def plot_eda_return_by_label():
    if not _ensure_dataset_loaded():
        return None
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(x='label', y='promised_return_pct', data=df, palette='pastel', ax=ax)
    ax.set_title('Promised Return Percentage by Label')
    fig.tight_layout()
    return fig


def plot_eda_correlation():
    if not _ensure_dataset_loaded():
        return None
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_cols = [
        'promised_return_pct', 'sentiment_score', 'message_length',
        'hour', 'dayofweek', 'is_weekend'
    ]
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
    ax.set_title("Correlation Heatmap of Features")
    fig.tight_layout()
    return fig


def plot_saved_image(image_path, title):
    if not os.path.exists(image_path):
        messagebox.showwarning("Graph Not Available", "Please train the model first to generate this graph.")
        return None
    fig, ax = plt.subplots(figsize=(9, 6))
    img = plt.imread(image_path)
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(title)
    fig.tight_layout()
    return fig


def ensure_graph_viewer():
    global graph_viewer_window, graph_canvas_container, graph_viewer_status_label
    if graph_viewer_window and graph_viewer_window.winfo_exists():
        return

    graph_viewer_window = tk.Toplevel(main)
    graph_viewer_window.title("Graph Viewer")
    graph_viewer_window.geometry("1100x700")
    graph_viewer_window.configure(bg=theme_value('window_bg'))
    graph_viewer_window.bind('<Left>', on_left_arrow)
    graph_viewer_window.bind('<Right>', on_right_arrow)

    viewer_outer = tk.Frame(graph_viewer_window, bg=theme_value('window_bg'))
    viewer_outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

    tk.Button(
        viewer_outer,
        text='<',
        command=show_previous_graph,
        font=('Segoe UI Bold', 15),
        width=3,
        bg=theme_value('action_bg'),
        fg=theme_value('action_fg'),
        relief='flat',
        cursor='hand2'
    ).pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))

    graph_canvas_container = tk.Frame(viewer_outer, bg='white', bd=1, relief='solid')
    graph_canvas_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    tk.Button(
        viewer_outer,
        text='>',
        command=show_next_graph,
        font=('Segoe UI Bold', 15),
        width=3,
        bg=theme_value('action_bg'),
        fg=theme_value('action_fg'),
        relief='flat',
        cursor='hand2'
    ).pack(side=tk.LEFT, fill=tk.Y, padx=(8, 0))

    graph_viewer_status_label = tk.Label(
        graph_viewer_window,
        text='',
        bg=theme_value('window_bg'),
        fg=theme_value('muted_fg'),
        font=('Segoe UI', 10)
    )
    graph_viewer_status_label.pack(pady=(0, 10))


def display_graph_in_viewer(fig, title_text):
    global graph_canvas_widget
    ensure_graph_viewer()

    if graph_canvas_widget is not None:
        graph_canvas_widget.get_tk_widget().destroy()
        graph_canvas_widget = None

    graph_canvas_widget = FigureCanvasTkAgg(fig, master=graph_canvas_container)
    graph_canvas_widget.draw()
    graph_canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    if graph_viewer_window and graph_viewer_window.winfo_exists():
        graph_viewer_window.title(f"Graph Viewer - {title_text}")
    if graph_viewer_status_label:
        graph_viewer_status_label.config(text=title_text)


def configure_graph_controls(plot_registry):
    global current_plot_registry, current_plot_index
    current_plot_registry = plot_registry
    current_plot_index = 0

    if current_plot_registry:
        selected_graph_name = current_plot_registry[0][0]
        graph_selected_label.config(text=f"Selected: {selected_graph_name}")
        graph_status_label.config(text=f"1 / {len(current_plot_registry)}")
    else:
        graph_selected_label.config(text='Selected: None')
        graph_status_label.config(text='0 / 0')


def show_graph_by_index(index):
    global current_plot_index
    if not current_plot_registry:
        return
    current_plot_index = index % len(current_plot_registry)
    selected_graph_name, plot_function = current_plot_registry[current_plot_index]
    graph_selected_label.config(text=f"Selected: {selected_graph_name}")
    graph_status_label.config(text=f"{current_plot_index + 1} / {len(current_plot_registry)}")
    fig = plot_function()
    if fig is not None:
        display_graph_in_viewer(fig, selected_graph_name)


def open_selected_graph(index):
    if not current_plot_registry:
        return
    show_graph_by_index(index)


def show_graph_dropdown(event=None):
    if not current_plot_registry:
        messagebox.showinfo("No Graphs Available", "No graphs available for the current role.")
        return

    graph_menu.delete(0, tk.END)
    for idx, (name, _) in enumerate(current_plot_registry):
        graph_menu.add_command(
            label=f"{idx + 1}. {name}",
            command=lambda i=idx: open_selected_graph(i)
        )

    x_position = graph_trigger_button.winfo_rootx()
    y_position = graph_trigger_button.winfo_rooty() + graph_trigger_button.winfo_height()
    graph_menu.tk_popup(x_position, y_position)
    graph_menu.grab_release()


def show_model_graph_dropdown(event, algorithm_name):
    filename_prefix = algorithm_name.replace(' ', '_')
    graph_options = []

    if os.path.isdir("results"):
        model_files = sorted(
            [f for f in os.listdir("results") if f.startswith(filename_prefix) and f.lower().endswith(".png")]
        )

        for image_name in model_files:
            image_path = os.path.join("results", image_name)
            readable_name = image_name.replace(filename_prefix + "_", "").replace(".png", "").replace("_", " ").title()
            graph_title = f"{algorithm_name} - {readable_name}"
            graph_options.append(
                (
                    readable_name,
                    lambda p=image_path, t=graph_title: open_model_graph_direct(p, t)
                )
            )

        # Keep shared comparison charts visible from all model buttons.
        shared_files = [
            ("Model Performance Comparison", "model_performance_comparison.png"),
            ("Model Performance Comparison (With Numbers)", "model_performance_comparison_with_numbers.png"),
        ]
        for label, shared_name in shared_files:
            shared_path = os.path.join("results", shared_name)
            if os.path.exists(shared_path):
                graph_options.append(
                    (
                        label,
                        lambda p=shared_path, t=label: open_model_graph_direct(p, t)
                    )
                )

    if not graph_options:
        graph_options = [
            (
                'Confusion Matrix',
                lambda: open_model_graph_direct(
                    f"results/{filename_prefix}_confusion_matrix.png",
                    f"{algorithm_name} - Confusion Matrix"
                )
            ),
            (
                'ROC Curve',
                lambda: open_model_graph_direct(
                    f"results/{filename_prefix}_roc_curve.png",
                    f"{algorithm_name} - ROC Curve"
                )
            ),
        ]

    menu = tk.Menu(main, tearoff=0, bg='white', fg='#102542', activebackground='#DDE8FF', activeforeground='#102542')
    for label, command in graph_options:
        menu.add_command(label=label, command=command)

    menu.tk_popup(event.widget.winfo_rootx(), event.widget.winfo_rooty() + event.widget.winfo_height())
    menu.grab_release()


def open_model_graph_direct(image_path, title):
    fig = plot_saved_image(image_path, title)
    if fig is not None:
        display_graph_in_viewer(fig, title)


def show_previous_graph():
    show_graph_by_index(current_plot_index - 1)


def show_next_graph():
    show_graph_by_index(current_plot_index + 1)


def on_left_arrow(event=None):
    if current_plot_registry:
        show_previous_graph()
        return "break"
    return None


def on_right_arrow(event=None):
    if current_plot_registry:
        show_next_graph()
        return "break"
    return None

def Train_Test_split():

    text.delete('1.0', END)
    global X, y, x_train, x_test, y_train, y_test
    if 'X' not in globals() or 'y' not in globals():
        messagebox.showerror("Read the Guide", "Read the Guide")
        return

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    text.insert(END, "Train-Test Split Completed\n")
    text.insert(END, "X_train shape: " + str(x_train.shape) + "\n")
    text.insert(END, "X_test shape : " + str(x_test.shape) + "\n")
    text.insert(END, "y_train shape: " + str(y_train.shape) + "\n")
    text.insert(END, "y_test shape : " + str(y_test.shape) + "\n")

classification_metrics_df = pd.DataFrame(
    columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
)
def calculate_metrics_tkinter(algorithm, y_pred, y_test, y_score=None):
    global classification_metrics_df, text
    text.delete('1.0', END)

    os.makedirs("results", exist_ok=True)

    # Load target categories
    encoder_path = os.path.join("models", "label_encoder.pkl")
    if os.path.exists(encoder_path):
        le_target = joblib.load(encoder_path)
        categories = le_target.classes_
    else:
        categories = ['legit', 'scam']

    # Compute metrics
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average='binary') * 100
    rec = recall_score(y_test, y_pred, average='binary') * 100
    f1 = f1_score(y_test, y_pred, average='binary') * 100

    # Append to global DataFrame
    classification_metrics_df.loc[len(classification_metrics_df)] = [
        algorithm, acc, prec, rec, f1
    ]

    text.insert(END, f"{algorithm} Metrics\n")
    text.insert(END, f"Accuracy : {acc:.2f}%\n")
    text.insert(END, f"Precision: {prec:.2f}%\n")
    text.insert(END, f"Recall   : {rec:.2f}%\n")
    text.insert(END, f"F1-Score : {f1:.2f}%\n\n")

    text.insert(END, "Classification Report:\n")
    text.insert(END, classification_report(
        y_test,
        y_pred,
        target_names=categories
    ))
    text.insert(END, "\n")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories
    )
    plt.title(f"{algorithm} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"results/{algorithm.replace(' ', '_')}_confusion_matrix.png")
    plt.close()

    # ROC Curve
    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{algorithm} - ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"results/{algorithm.replace(' ', '_')}_roc_curve.png")
        plt.close()

      
def train_xgboost_classifier_tkinter(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators=100,
    learning_rate=0.01,
    max_depth=6,
    subsample=1.0,
    colsample_bytree=1.0
):
    text.delete('1.0', END)

    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    text.delete('1.0', END)
    model_path = os.path.join(MODEL_DIR, 'xgboost.pkl')

    if os.path.exists(model_path):
        text.insert(END, "Loading XGBoost Classifier...\n")
        model = joblib.load(model_path)
    else:
        text.insert(END, "Training XGBoost Classifier...\n")
        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, f"Model saved to {model_path}\n")

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    calculate_metrics_tkinter(
        algorithm="XGBoost Classifier",
        y_pred=y_pred,
        y_test=y_test,
        y_score=y_score
    )

    return model


def train_lightgbm_classifier_tkinter(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators=100,
    learning_rate=0.01,
    max_depth=-1,
    num_leaves=31,
    subsample=1.0,
    colsample_bytree=1.0
):

    text.delete('1.0', END)

    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    text.delete('1.0', END)
    model_path = os.path.join(MODEL_DIR, 'lightgbm.pkl')

    if os.path.exists(model_path):
        text.insert(END, "Loading LightGBM Classifier...\n")
        model = joblib.load(model_path)
    else:
        text.insert(END, "Training LightGBM Classifier...\n")
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42
        )
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, f"Model saved to {model_path}\n")

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    calculate_metrics_tkinter(
        algorithm="LightGBM Classifier",
        y_pred=y_pred,
        y_test=y_test,
        y_score=y_score
    )

    return model

    
def train_adaboost_classifier_tkinter(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators=50,
    learning_rate=0.01
):
    text.delete('1.0', END)

    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    text.delete('1.0', END)
    model_path = os.path.join(MODEL_DIR, 'adaboost.pkl')

    if os.path.exists(model_path):
        text.insert(END, "Loading AdaBoost Classifier...\n")
        model = joblib.load(model_path)
    else:
        text.insert(END, "Training AdaBoost Classifier...\n")
        model = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, f"Model saved to {model_path}\n")

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    calculate_metrics_tkinter(
        algorithm="AdaBoost Classifier",
        y_pred=y_pred,
        y_test=y_test,
        y_score=y_score
    )

    return model

def train_stacking_sgd_pac_tkinter(
    X_train,
    y_train,
    X_test,
    y_test,
    final_estimator=None
):
    text.delete('1.0', END)

    MODEL_DIR = "models"
    os.makedirs(MODEL_DIR, exist_ok=True)
    text.delete('1.0', END)
    model_path = os.path.join(MODEL_DIR, 'stacking_sgd_pac.pkl')

    if os.path.exists(model_path):
        text.insert(END, "Loading Stacking Classifier (SGD + PAC)...\n")
        model = joblib.load(model_path)
    else:
        text.insert(END, "Training Stacking Classifier (SGD + PAC)...\n")

        estimators = [
            ('sgd', SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, tol=1e-3, random_state=42)),
            ('pac', PassiveAggressiveClassifier(C=1.0, max_iter=1000, tol=1e-3, random_state=42))
        ]

        if final_estimator is None:
            from sklearn.linear_model import LogisticRegression
            final_estimator = LogisticRegression()

        model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            n_jobs=-1,
            passthrough=False
        )

        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        text.insert(END, f"Model saved to {model_path}\n")

    y_pred = model.predict(X_test)
    try:
        y_score = model.predict_proba(X_test)
    except:
        y_score = None

    calculate_metrics_tkinter(
        algorithm="Stacking Classifier (SGD + PAC)",
        y_pred=y_pred,
        y_test=y_test,
        y_score=y_score
    )

    return model

def plot_model_performance_tkinter():
    text.delete('1.0', END)
    text.insert(END, "Plotting Model Performance Comparison...\n")

    if classification_metrics_df.empty:
        text.insert(END, "No model metrics found.\n")
        text.insert(END, "Train at least one model first, then open this graph.\n")
        messagebox.showinfo(
            "No Metrics Available",
            "No model metrics to plot yet.\nPlease train one or more models first."
        )
        return None

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))

    df_melt = classification_metrics_df.melt(
        id_vars='Algorithm',
        value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        var_name='Metric',
        value_name='Score'
    )

    ax = sns.barplot(x='Algorithm', y='Score', hue='Metric', data=df_melt, ax=ax)

    ax.tick_params(axis='x', rotation=45)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.1f}", 
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 2),
                    textcoords='offset points')

    ax.legend(title="Metric")
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/model_performance_comparison.png")

    text.insert(END, "Plot saved as 'results/model_performance_comparison.png'\n")
    return fig

    
def predict_testdata_tkinter():
    text.delete('1.0', END)
    try:
        file_path = filedialog.askopenfilename(
            initialdir=".",
            title="Select Test CSV File",
            filetypes=(("CSV Files", "*.csv"),)
        )
        if not file_path:
            text.insert(END, "No file selected.\n")
            return

        test_df = pd.read_csv(file_path)
        text.insert(END, f"Test dataset loaded: {test_df.shape[0]} rows\n")

        W2V_PATH = os.path.join("models", "word2vec.model")
        if os.path.exists(W2V_PATH):
            w2v_model = Word2Vec.load(W2V_PATH)
            text.insert(END, "Word2Vec model loaded for prediction.\n")
        else:
            text.insert(END, "Word2Vec model not found! Please preprocess/train first.\n")
            return

        X_testdata, _, _ = preprocess_data_word2vec(
            test_df,
            is_train=False,
            w2v_model=w2v_model,
            apply_smote=False
        )

        stack_path = os.path.join(MODEL_DIR, 'stacking_sgd_pac.pkl')
        stack_model = joblib.load(stack_path)

        stack_preds = stack_model.predict(X_testdata)
        le_target = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        stack_labels = le_target.inverse_transform(stack_preds)
        test_df['predictions'] = stack_labels

        text.insert(END, "Predictions completed.\n")
        text.insert(END, f"Predictions:\n {test_df}\n")

    except Exception as e:
        text.insert(END, f"Error: {str(e)}\n")


def close():
    main.destroy()
    

# Predefined credentials
ADMIN_CREDENTIALS = {"username": "admin", "password": "admin"}
USER_CREDENTIALS  = {"username": "user", "password": "user"}


def center_window(win, width, height):
    win.update_idletasks()
    x_coord = (win.winfo_screenwidth() // 2) - (width // 2)
    y_coord = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry(f"{width}x{height}+{x_coord}+{y_coord}")


def theme_value(key):
    return THEMES[current_theme][key]


def apply_theme():
    main.config(bg=theme_value('window_bg'))
    header_frame.config(bg=theme_value('window_bg'))
    title.config(bg=theme_value('header_bg'), fg=theme_value('header_fg'))

    top_controls_frame.config(bg=theme_value('window_bg'))
    role_status_label.config(bg=theme_value('window_bg'), fg=theme_value('panel_fg'))
    theme_toggle_button.config(
        bg=theme_value('button_bg'),
        fg=theme_value('button_fg'),
        activebackground=theme_value('button_active_bg'),
        activeforeground=theme_value('button_fg')
    )
    guide_button.config(
        bg=theme_value('button_bg'),
        fg=theme_value('button_fg'),
        activebackground=theme_value('button_active_bg'),
        activeforeground=theme_value('button_fg')
    )
    logout_button.config(
        bg=theme_value('button_bg'),
        fg=theme_value('button_fg'),
        activebackground=theme_value('button_active_bg'),
        activeforeground=theme_value('button_fg')
    )

    graph_frame.config(bg=theme_value('panel_bg'))
    graph_title_label.config(bg=theme_value('panel_bg'), fg=theme_value('panel_fg'))
    graph_trigger_button.config(
        bg=theme_value('button_bg'),
        fg=theme_value('button_fg'),
        activebackground=theme_value('button_active_bg'),
        activeforeground=theme_value('button_fg')
    )
    graph_prev_button.config(bg=theme_value('action_bg'), fg=theme_value('action_fg'))
    graph_next_button.config(bg=theme_value('action_bg'), fg=theme_value('action_fg'))
    graph_selected_label.config(bg=theme_value('panel_bg'), fg=theme_value('muted_fg'))
    graph_status_label.config(bg=theme_value('panel_bg'), fg=theme_value('muted_fg'))

    output_frame.config(bg=theme_value('window_bg'))
    text.config(bg=theme_value('output_bg'), fg=theme_value('output_fg'), insertbackground=theme_value('output_fg'))

    action_frame.config(bg=theme_value('window_bg'))
    for widget in action_frame.winfo_children():
        if isinstance(widget, tk.Button):
            widget.config(
                bg=theme_value('action_bg'),
                fg=theme_value('action_fg'),
                activebackground=theme_value('action_active_bg'),
                activeforeground=theme_value('action_fg')
            )

    if login_screen_frame and login_screen_frame.winfo_exists():
        login_screen_frame.config(bg=theme_value('window_bg'))
        login_card.config(bg=theme_value('card_bg'))
        login_heading_label.config(bg=theme_value('card_bg'), fg=theme_value('panel_fg'))
        login_hint_label.config(bg=theme_value('card_bg'), fg=theme_value('muted_fg'))
        login_admin_button.config(bg='#9AD0C2', fg='#0B3D2E')
        login_user_button.config(bg='#FFD6A5', fg='#5A3000')
        login_guide_button.config(
            bg=theme_value('button_bg'),
            fg=theme_value('button_fg'),
            activebackground=theme_value('button_active_bg'),
            activeforeground=theme_value('button_fg')
        )
        login_theme_button.config(
            bg=theme_value('button_bg'),
            fg=theme_value('button_fg'),
            activebackground=theme_value('button_active_bg'),
            activeforeground=theme_value('button_fg')
        )

    if graph_viewer_window and graph_viewer_window.winfo_exists():
        graph_viewer_window.config(bg=theme_value('window_bg'))
        if graph_viewer_status_label and graph_viewer_status_label.winfo_exists():
            graph_viewer_status_label.config(bg=theme_value('window_bg'), fg=theme_value('muted_fg'))

    if active_login_window and active_login_window.winfo_exists():
        active_login_window.config(bg=theme_value('window_bg'))


def toggle_theme():
    global current_theme
    current_theme = 'dark' if current_theme == 'light' else 'light'
    theme_toggle_button.config(text=f"Theme: {current_theme.title()}")
    login_theme_button.config(text=f"Theme: {current_theme.title()}")
    apply_theme()
    main.update_idletasks()


def show_login_screen():
    for section in dashboard_sections:
        section.pack_forget()
    login_screen_frame.pack(fill=tk.BOTH, expand=True)


def show_dashboard():
    login_screen_frame.pack_forget()
    header_frame.pack(fill=tk.X, padx=14, pady=(14, 8))
    top_controls_frame.pack(fill=tk.X, padx=14, pady=(2, 8))
    action_frame.pack(fill=tk.X, padx=14, pady=6)
    graph_frame.pack(fill=tk.X, padx=14, pady=(6, 10))
    output_frame.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))


def logout():
    global current_role, current_plot_registry, current_plot_index, graph_canvas_widget, active_login_window
    current_role = None
    clear_buttons()
    current_plot_registry = []
    current_plot_index = 0
    configure_graph_controls([])
    role_status_label.config(text='Not logged in')
    text.delete('1.0', END)
    if graph_viewer_window and graph_viewer_window.winfo_exists():
        graph_viewer_window.destroy()
    graph_canvas_widget = None
    if active_login_window and active_login_window.winfo_exists():
        active_login_window.destroy()
    active_login_window = None
    show_login_screen()
    apply_theme()


def require_training_ready():
    required = ['df', 'X', 'y', 'x_train', 'x_test', 'y_train', 'y_test']
    for item in required:
        if item not in globals():
            messagebox.showerror("Read the Guide", "Read the Guide")
            return False
    return True
def authenticate(role):
    global active_login_window
    login_win = tk.Toplevel(main)
    active_login_window = login_win
    login_win.title(f"{role} Login")
    login_win.configure(bg=theme_value('window_bg'))
    login_win.resizable(False, False)
    center_window(login_win, 400, 320)
    login_win.grab_set()
    login_win.transient(main)
    login_win.protocol("WM_DELETE_WINDOW", lambda: _close_login_window(login_win))

    login_card = tk.Frame(login_win, bg=theme_value('card_bg'), bd=1, relief='solid')
    login_card.place(relx=0.5, rely=0.5, anchor='center', width=320, height=250)

    tk.Label(
        login_card,
        text=f"{role} Login",
        bg=theme_value('card_bg'),
        fg=theme_value('panel_fg'),
        font=('Segoe UI Semibold', 14)
    ).pack(pady=(16, 10))

    tk.Label(login_card, text="Username", bg=theme_value('card_bg'), fg=theme_value('muted_fg'), font=('Segoe UI', 10)).pack(anchor='w', padx=30)
    username_entry = tk.Entry(login_card, font=('Segoe UI', 10), relief='solid', bd=1)
    username_entry.pack(fill=tk.X, padx=30, pady=(4, 10))

    tk.Label(login_card, text="Password", bg=theme_value('card_bg'), fg=theme_value('muted_fg'), font=('Segoe UI', 10)).pack(anchor='w', padx=30)
    password_entry = tk.Entry(login_card, show="*", font=('Segoe UI', 10), relief='solid', bd=1)
    password_entry.pack(fill=tk.X, padx=30, pady=(4, 14))

    def check_login():
        global active_login_window
        username = username_entry.get()
        password = password_entry.get()

        if role == "ADMIN":
            if username == ADMIN_CREDENTIALS["username"] and password == ADMIN_CREDENTIALS["password"]:
                login_win.destroy()
                active_login_window = None
                complete_login("ADMIN")
            else:
                messagebox.showerror("Error", "Invalid Admin credentials!")
        elif role == "USER":
            if username == USER_CREDENTIALS["username"] and password == USER_CREDENTIALS["password"]:
                login_win.destroy()
                active_login_window = None
                complete_login("USER")
            else:
                messagebox.showerror("Error", "Invalid User credentials!")

    tk.Button(
        login_card,
        text="Login",
        command=check_login,
        font=('Segoe UI Semibold', 10),
        bg=theme_value('button_bg'),
        fg=theme_value('button_fg'),
        activebackground=theme_value('button_active_bg'),
        activeforeground=theme_value('button_fg'),
        relief='flat',
        cursor='hand2',
        padx=10,
        pady=6
    ).pack(pady=2)

    username_entry.focus_set()
    login_win.bind('<Return>', lambda event: check_login())


def complete_login(role):
    global current_role
    current_role = role
    role_status_label.config(text=f"Signed in as: {role}")
    show_dashboard()
    if role == "ADMIN":
        show_admin_buttons()
    else:
        show_user_buttons()
    apply_theme()


def _close_login_window(login_win):
    global active_login_window
    if login_win and login_win.winfo_exists():
        login_win.destroy()
    active_login_window = None


def show_guide():
    guide_text = (
        "Recommended App Flow (Use This Order)\n\n"
        "Admin Flow:\n"
        "1. Login as ADMIN.\n"
        "2. Click Upload Dataset FIRST.\n"
        "3. Click Preprocess Dataset.\n"
        "4. Click Dataset Splitting.\n"
        "5. Then click Train XGBoost / Train LightGBM / Train AdaBoost / Train Stacking.\n"
        "6. Hover on training buttons to open model-specific graph list.\n"
        "7. Use Flash Graphs for EDA and comparison charts.\n\n"
        "User Flow:\n"
        "1. Login as USER.\n"
        "2. Click Predict on Test Data.\n"
        "3. Use Flash Graphs to open model performance graph.\n\n"
        "Important:\n"
        "- If training is clicked before upload/preprocess/split, app will show: Read the Guide.\n\n"
        "Graph Controls:\n"
        "- Hover over Flash Graphs to see available plots.\n"
        "- Hover over model training buttons to see model graph options.\n"
        "- Select a graph from the dropdown to open it.\n"
        "- Use < and > for quick previous/next navigation."
    )

    guide_win = tk.Toplevel(main)
    guide_win.title("Usage Guide")
    guide_win.configure(bg=theme_value('window_bg'))
    guide_win.resizable(False, False)
    center_window(guide_win, 520, 430)
    guide_win.transient(main)

    guide_card = tk.Frame(guide_win, bg=theme_value('card_bg'), bd=1, relief='solid')
    guide_card.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)

    tk.Label(
        guide_card,
        text='How To Use This Dashboard',
        bg=theme_value('card_bg'),
        fg=theme_value('panel_fg'),
        font=('Segoe UI Semibold', 14)
    ).pack(pady=(14, 10))

    guide_box = Text(
        guide_card,
        bg=theme_value('card_bg'),
        fg=theme_value('panel_fg'),
        font=('Segoe UI', 10),
        wrap='word',
        relief='flat',
        padx=14,
        pady=8
    )
    guide_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    guide_box.insert(END, guide_text)
    guide_box.config(state='disabled')


def show_admin_buttons():
    clear_buttons()
    create_action_button('Upload Dataset', uploadDataset)
    create_action_button('Preprocess Dataset', lambda: preprocess_data_word2vec(df) if 'df' in globals() else messagebox.showerror("Read the Guide", "Read the Guide"))
    create_action_button('Dataset Splitting', Train_Test_split)
    create_model_action_button(
        'Train XGBoost',
        lambda: train_xgboost_classifier_tkinter(x_train, y_train, x_test, y_test) if require_training_ready() else None,
        'XGBoost Classifier'
    )
    create_model_action_button(
        'Train LightGBM',
        lambda: train_lightgbm_classifier_tkinter(x_train, y_train, x_test, y_test) if require_training_ready() else None,
        'LightGBM Classifier'
    )
    create_model_action_button(
        'Train AdaBoost',
        lambda: train_adaboost_classifier_tkinter(x_train, y_train, x_test, y_test) if require_training_ready() else None,
        'AdaBoost Classifier'
    )
    create_model_action_button(
        'Train Stacking SGD_PAC',
        lambda: train_stacking_sgd_pac_tkinter(x_train, y_train, x_test, y_test) if require_training_ready() else None,
        'Stacking Classifier (SGD + PAC)'
    )
    configure_graph_controls([
        ('Distribution: Scam vs Legit', plot_eda_distribution),
        ('Platform vs Scam/Legit', plot_eda_platform),
        ('Sentiment vs Message Length', plot_eda_sentiment_vs_length),
        ('Promised Return by Label', plot_eda_return_by_label),
        ('Feature Correlation Heatmap', plot_eda_correlation),
        ('Model Performance Comparison', plot_model_performance_tkinter),
    ])


def show_user_buttons():
    clear_buttons()
    create_action_button('Predict on Test Data', predict_testdata_tkinter)
    create_action_button('Logout', logout)
    configure_graph_controls([
        ('Model Performance Comparison', plot_model_performance_tkinter),
    ])


def clear_buttons():
    for widget in action_frame.winfo_children():
        widget.destroy()


def create_action_button(label, command):
    btn = tk.Button(
        action_frame,
        text=label,
        command=command,
        font=button_font,
        width=22,
        bg='#E8F1FF',
        fg='#102542',
        activebackground='#C6DDFF',
        activeforeground='#102542',
        relief='flat',
        padx=10,
        pady=8,
        cursor='hand2'
    )
    btn.pack(side=tk.LEFT, padx=6, pady=6)
    return btn


def create_model_action_button(label, command, algorithm_name):
    btn = create_action_button(label, command)
    btn.bind('<Enter>', lambda event, algo=algorithm_name: show_model_graph_dropdown(event, algo))
    return btn


main = tk.Tk()
main.title('Bitcoin Scam Detection')
main.geometry('1240x760')
main.config(bg=theme_value('window_bg'))


def scroll_title(text, label, delay=200):
    def shift():
        nonlocal text
        text = text[1:] + text[0]
        label.config(text=text)
        label.after(delay, shift)
    shift()


title_font = ('Segoe UI Semibold', 20)
button_font = ('Segoe UI Semibold', 10)
text_font = ('Consolas', 10)
label_font = ('Segoe UI', 10)

login_screen_frame = tk.Frame(main, bg=theme_value('window_bg'))
login_card = tk.Frame(login_screen_frame, bg=theme_value('card_bg'), bd=1, relief='solid')
login_card.place(relx=0.5, rely=0.5, anchor='center', width=460, height=300)

login_heading_label = tk.Label(
    login_card,
    text='Bitcoin Scam Detection',
    bg=theme_value('card_bg'),
    fg=theme_value('panel_fg'),
    font=('Segoe UI Semibold', 18)
)
login_heading_label.pack(pady=(20, 10))

login_hint_label = tk.Label(
    login_card,
    text='Login as Admin or User to open the dashboard',
    bg=theme_value('card_bg'),
    fg=theme_value('muted_fg'),
    font=('Segoe UI', 10)
)
login_hint_label.pack(pady=(0, 14))

login_buttons_row = tk.Frame(login_card, bg=theme_value('card_bg'))
login_buttons_row.pack(pady=(0, 10))

login_admin_button = tk.Button(
    login_buttons_row,
    text='ADMIN LOGIN',
    command=lambda: authenticate('ADMIN'),
    font=('Segoe UI Semibold', 10),
    width=16,
    bg='#9AD0C2',
    fg='#0B3D2E',
    relief='flat',
    cursor='hand2',
    pady=8
)
login_admin_button.pack(side=tk.LEFT, padx=(0, 8))

login_user_button = tk.Button(
    login_buttons_row,
    text='USER LOGIN',
    command=lambda: authenticate('USER'),
    font=('Segoe UI Semibold', 10),
    width=16,
    bg='#FFD6A5',
    fg='#5A3000',
    relief='flat',
    cursor='hand2',
    pady=8
)
login_user_button.pack(side=tk.LEFT, padx=(8, 0))

login_helpers_row = tk.Frame(login_card, bg=theme_value('card_bg'))
login_helpers_row.pack(pady=(8, 0))

login_guide_button = tk.Button(
    login_helpers_row,
    text='Guide',
    command=show_guide,
    font=button_font,
    bg=theme_value('button_bg'),
    fg=theme_value('button_fg'),
    activebackground=theme_value('button_active_bg'),
    activeforeground=theme_value('button_fg'),
    relief='flat',
    cursor='hand2',
    padx=16,
    pady=6
)
login_guide_button.pack(side=tk.LEFT, padx=(0, 8))

login_theme_button = tk.Button(
    login_helpers_row,
    text=f"Theme: {current_theme.title()}",
    command=toggle_theme,
    font=button_font,
    bg=theme_value('button_bg'),
    fg=theme_value('button_fg'),
    activebackground=theme_value('button_active_bg'),
    activeforeground=theme_value('button_fg'),
    relief='flat',
    cursor='hand2',
    padx=12,
    pady=6
)
login_theme_button.pack(side=tk.LEFT)

header_frame = tk.Frame(main, bg=theme_value('window_bg'))

title = Label(
    header_frame,
    text='Bitcoin Scam Detection',
    bg=theme_value('header_bg'),
    fg=theme_value('header_fg'),
    font=title_font,
    pady=12,
    padx=12
)
title.pack(fill=tk.X)

scroll_title('   Bitcoin Scam Detection Dashboard   ', title, delay=200)

top_controls_frame = tk.Frame(main, bg=theme_value('window_bg'))
role_status_label = tk.Label(
    top_controls_frame,
    text='Not logged in',
    bg=theme_value('window_bg'),
    fg=theme_value('panel_fg'),
    font=('Segoe UI Semibold', 10)
)
role_status_label.pack(side=tk.LEFT)

theme_toggle_button = tk.Button(
    top_controls_frame,
    text=f"Theme: {current_theme.title()}",
    command=toggle_theme,
    font=button_font,
    bg=theme_value('button_bg'),
    fg=theme_value('button_fg'),
    activebackground=theme_value('button_active_bg'),
    activeforeground=theme_value('button_fg'),
    relief='flat',
    cursor='hand2',
    padx=12,
    pady=5
)
theme_toggle_button.pack(side=tk.RIGHT, padx=(8, 0))

guide_button = tk.Button(
    top_controls_frame,
    text='Guide',
    command=show_guide,
    font=button_font,
    bg=theme_value('button_bg'),
    fg=theme_value('button_fg'),
    activebackground=theme_value('button_active_bg'),
    activeforeground=theme_value('button_fg'),
    relief='flat',
    cursor='hand2',
    padx=12,
    pady=5
)
guide_button.pack(side=tk.RIGHT, padx=(8, 0))

logout_button = tk.Button(
    top_controls_frame,
    text='Logout',
    command=logout,
    font=button_font,
    bg=theme_value('button_bg'),
    fg=theme_value('button_fg'),
    activebackground=theme_value('button_active_bg'),
    activeforeground=theme_value('button_fg'),
    relief='flat',
    cursor='hand2',
    padx=12,
    pady=5
)
logout_button.pack(side=tk.RIGHT, padx=(8, 0))

action_frame = tk.Frame(main, bg=theme_value('window_bg'))

graph_frame = tk.Frame(main, bg=theme_value('panel_bg'), bd=1, relief='solid')

graph_title_label = tk.Label(
    graph_frame,
    text='Graph Controls',
    font=('Segoe UI Semibold', 11),
    bg=theme_value('panel_bg'),
    fg=theme_value('panel_fg')
)
graph_title_label.pack(side=tk.LEFT, padx=(10, 8), pady=8)

graph_trigger_button = tk.Button(
    graph_frame,
    text='Flash Graphs',
    command=show_graph_dropdown,
    font=('Segoe UI Semibold', 10),
    bg=theme_value('button_bg'),
    fg=theme_value('button_fg'),
    activebackground=theme_value('button_active_bg'),
    activeforeground=theme_value('button_fg'),
    relief='flat',
    cursor='hand2',
    padx=12,
    pady=6
)
graph_trigger_button.pack(side=tk.LEFT, padx=(0, 10), pady=8)
graph_trigger_button.bind('<Enter>', show_graph_dropdown)

graph_menu = tk.Menu(main, tearoff=0, bg='white', fg='#102542', activebackground='#DDE8FF', activeforeground='#102542')

graph_prev_button = tk.Button(
    graph_frame,
    text='<',
    command=show_previous_graph,
    font=('Segoe UI Bold', 11),
    width=3,
    bg=theme_value('action_bg'),
    fg=theme_value('action_fg'),
    relief='flat',
    cursor='hand2'
)
graph_prev_button.pack(side=tk.LEFT, padx=(10, 4), pady=8)

graph_next_button = tk.Button(
    graph_frame,
    text='>',
    command=show_next_graph,
    font=('Segoe UI Bold', 11),
    width=3,
    bg=theme_value('action_bg'),
    fg=theme_value('action_fg'),
    relief='flat',
    cursor='hand2'
)
graph_next_button.pack(side=tk.LEFT, padx=(0, 8), pady=8)

graph_selected_label = tk.Label(graph_frame, text='Selected: None', font=label_font, bg=theme_value('panel_bg'), fg=theme_value('muted_fg'))
graph_selected_label.pack(side=tk.LEFT, padx=4, pady=8)

graph_status_label = tk.Label(graph_frame, text='0 / 0', font=label_font, bg=theme_value('panel_bg'), fg=theme_value('muted_fg'))
graph_status_label.pack(side=tk.LEFT, padx=4, pady=8)

output_frame = tk.Frame(main, bg=theme_value('window_bg'))

text = Text(
    output_frame,
    bg=theme_value('output_bg'),
    fg=theme_value('output_fg'),
    insertbackground=theme_value('output_fg'),
    font=text_font,
    wrap='word',
    relief='flat',
    padx=8,
    pady=8
)
scroll = Scrollbar(output_frame)
text.configure(yscrollcommand=scroll.set)
scroll.config(command=text.yview)
text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scroll.pack(side=tk.RIGHT, fill=tk.Y)

dashboard_sections = [header_frame, top_controls_frame, action_frame, graph_frame, output_frame]
configure_graph_controls([])
apply_theme()
show_login_screen()
main.bind('<Left>', on_left_arrow)
main.bind('<Right>', on_right_arrow)

main.mainloop()
