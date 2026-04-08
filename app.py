import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, LogisticRegression
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

warnings.filterwarnings("ignore")

MODEL_DIR = "models"
RESULTS_DIR = "results"
W2V_DIM = 100
W2V_PATH = os.path.join(MODEL_DIR, "word2vec.model")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

ADMIN_CREDENTIALS = {"username": "admin", "password": "admin"}
USER_REGISTRY = {"user": "user"}

FLASH_GRAPH_NAMES = [
    "Distribution: Scam vs Legit",
    "Platform vs Scam/Legit",
    "Sentiment vs Message Length",
    "Promised Return by Label",
    "Feature Correlation Heatmap",
    "Model Performance Comparison",
]

MODEL_TO_ALGO = {
    "XGBoost": "XGBoost Classifier",
    "LightGBM": "LightGBM Classifier",
    "AdaBoost": "AdaBoost Classifier",
    "Stacking SGD_PAC": "Stacking Classifier (SGD + PAC)",
}

classification_metrics_df = pd.DataFrame(
    columns=["Algorithm", "Accuracy", "Precision", "Recall", "F1-Score"]
)


def init_session():
    return {
        "logged_in": False,
        "role": None,
        "df": None,
        "X": None,
        "y": None,
        "x_train": None,
        "x_test": None,
        "y_train": None,
        "y_test": None,
        "graph_index": 0,
    }


def require_role(session, allowed):
    if not session.get("logged_in"):
        return False, "Please login first."
    if session.get("role") not in allowed:
        return False, "Access denied for this role."
    return True, "OK"


def login(role, username, password, session):
    session = session or init_session()
    if role == "ADMIN":
        valid = username == ADMIN_CREDENTIALS["username"] and password == ADMIN_CREDENTIALS["password"]
    else:
        valid = USER_REGISTRY.get(username) == password
    if valid:
        session["logged_in"] = True
        session["role"] = role
        return session, f"Logged in as {role}"
    return session, "Invalid credentials."


def signup(username, password, confirm_password):
    if not username or not password:
        return "Username and password are required."
    if username.lower() == ADMIN_CREDENTIALS["username"]:
        return "This username is reserved."
    if password != confirm_password:
        return "Passwords do not match."
    if username in USER_REGISTRY:
        return "Username already exists. Please login."
    USER_REGISTRY[username] = password
    return "Signup successful. You can now login as USER."


def logout(session):
    return init_session(), "Logged out. Please login again."


def show_guide_content():
    return gr.update(visible=True)


def handle_login(role, username, password, session):
    session, status = login(role, username, password, session)
    if session.get("logged_in"):
        return (
            session,
            status,
            gr.update(visible=False),
            gr.update(visible=True),
            f"Welcome, {session['role']}",
        )
    return (
        session,
        status,
        gr.update(visible=True),
        gr.update(visible=False),
        "Not logged in",
    )


def handle_logout(session):
    session, status = logout(session)
    return (
        session,
        status,
        gr.update(visible=True),
        gr.update(visible=False),
        "Not logged in",
    )


def upload_dataset(file_obj, session):
    ok, msg = require_role(session, {"ADMIN"})
    if not ok:
        return session, msg, None
    if file_obj is None:
        return session, "Please upload a CSV dataset.", None
    df = pd.read_csv(file_obj)
    session["df"] = df
    session["X"] = None
    session["y"] = None
    session["x_train"] = None
    session["x_test"] = None
    session["y_train"] = None
    session["y_test"] = None
    return session, f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.", df.head(20)


def _transform_with_unknown(le, series):
    values = series.astype(str)
    known = set(le.classes_)
    fallback = le.classes_[0]
    safe = values.apply(lambda x: x if x in known else fallback)
    return le.transform(safe)


def train_word2vec(sentences, dim=W2V_DIM):
    tokenized = [str(s).split() for s in sentences]
    model = Word2Vec(
        sentences=tokenized,
        vector_size=dim,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
    )
    model.save(W2V_PATH)
    return model


def sentence_to_word2vec(sentence, model, dim=W2V_DIM):
    words = str(sentence).split()
    vecs = [model.wv[w] for w in words if w in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(dim)


def preprocess_data_word2vec(df, is_train=True, w2v_model=None, apply_smote=True, random_state=42):
    df = df.copy()
    logs = ["Starting preprocessing..."]

    df.drop(columns=["scam_type"], errors="ignore", inplace=True)
    target_col = "label"
    y = None
    le_target_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

    if is_train and target_col in df.columns:
        le_target = LabelEncoder()
        df[target_col] = le_target.fit_transform(df[target_col].astype(str))
        joblib.dump(le_target, le_target_path)
        y = df[target_col].values
    elif (not is_train) and target_col in df.columns and os.path.exists(le_target_path):
        le_target = joblib.load(le_target_path)
        df[target_col] = _transform_with_unknown(le_target, df[target_col])
        y = df[target_col].values

    categorical_cols = ["platform", "urgency_level", "contains_link", "btc_address_present"]
    for col in categorical_cols:
        encoder_path = os.path.join(MODEL_DIR, f"{col}_encoder.pkl")
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            joblib.dump(le, encoder_path)
        else:
            if not os.path.exists(encoder_path):
                raise ValueError(f"Missing encoder for {col}. Train preprocessing first.")
            le = joblib.load(encoder_path)
            df[col] = _transform_with_unknown(le, df[col])

    numeric_cols = ["promised_return_pct", "sentiment_score", "message_length", "hour", "dayofweek", "is_weekend"]
    X_structured = df[categorical_cols + numeric_cols].copy()
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    if is_train:
        scaler = StandardScaler()
        X_structured[numeric_cols] = scaler.fit_transform(X_structured[numeric_cols])
        joblib.dump(scaler, scaler_path)
    else:
        if not os.path.exists(scaler_path):
            raise ValueError("Missing scaler. Train preprocessing first.")
        scaler = joblib.load(scaler_path)
        X_structured[numeric_cols] = scaler.transform(X_structured[numeric_cols])

    if is_train:
        w2v_model = train_word2vec(df["message_text"])
    elif w2v_model is None:
        if not os.path.exists(W2V_PATH):
            raise ValueError("Missing Word2Vec model. Train preprocessing first.")
        w2v_model = Word2Vec.load(W2V_PATH)

    X_w2v = np.vstack(df["message_text"].apply(lambda x: sentence_to_word2vec(x, w2v_model)))
    X = np.hstack([X_w2v, X_structured.values])
    logs.append(f"Combined features shape: {X.shape}")

    if is_train and apply_smote and y is not None:
        smote = SMOTE(random_state=random_state)
        X, y = smote.fit_resample(X, y)
        logs.append(f"SMOTE applied. Samples: {len(y)}")

    logs.append("Preprocessing completed.")
    return X, y, "\n".join(logs)


def preprocess_dataset(session):
    ok, msg = require_role(session, {"ADMIN"})
    if not ok:
        return session, msg
    if session.get("df") is None:
        return session, "Read the Guide"
    X, y, logs = preprocess_data_word2vec(session["df"], is_train=True, apply_smote=True)
    session["X"] = X
    session["y"] = y
    return session, logs


def split_dataset(session):
    ok, msg = require_role(session, {"ADMIN"})
    if not ok:
        return session, msg
    if session.get("X") is None or session.get("y") is None:
        return session, "Read the Guide"
    x_train, x_test, y_train, y_test = train_test_split(
        session["X"], session["y"], test_size=0.20, random_state=42, stratify=session["y"]
    )
    session["x_train"] = x_train
    session["x_test"] = x_test
    session["y_train"] = y_train
    session["y_test"] = y_test
    return session, f"Split done. X_train: {x_train.shape}, X_test: {x_test.shape}"


def calculate_metrics(algorithm, y_pred, y_test, y_score=None):
    global classification_metrics_df
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, average="binary") * 100
    rec = recall_score(y_test, y_pred, average="binary") * 100
    f1 = f1_score(y_test, y_pred, average="binary") * 100

    classification_metrics_df.loc[len(classification_metrics_df)] = [algorithm, acc, prec, rec, f1]

    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    if os.path.exists(encoder_path):
        categories = joblib.load(encoder_path).classes_
    else:
        categories = ["legit", "scam"]

    report = [
        f"{algorithm} Metrics",
        f"Accuracy : {acc:.2f}%",
        f"Precision: {prec:.2f}%",
        f"Recall   : {rec:.2f}%",
        f"F1-Score : {f1:.2f}%",
        "",
        "Classification Report:",
        classification_report(y_test, y_pred, target_names=categories),
    ]

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)
    plt.title(f"{algorithm} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{algorithm.replace(' ', '_')}_confusion_matrix.png"))
    plt.close()

    if y_score is not None:
        fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"{algorithm} - ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{algorithm.replace(' ', '_')}_roc_curve.png"))
        plt.close()

    return "\n".join(report)


def _train_model(session, algorithm):
    ok, msg = require_role(session, {"ADMIN"})
    if not ok:
        return session, msg
    if any(session.get(k) is None for k in ["x_train", "x_test", "y_train", "y_test"]):
        return session, "Read the Guide"

    x_train, x_test = session["x_train"], session["x_test"]
    y_train, y_test = session["y_train"], session["y_test"]

    if algorithm == "XGBoost Classifier":
        model_path = os.path.join(MODEL_DIR, "xgboost.pkl")
        model = joblib.load(model_path) if os.path.exists(model_path) else XGBClassifier(
            n_estimators=100,
            learning_rate=0.01,
            max_depth=6,
            subsample=1.0,
            colsample_bytree=1.0,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        ).fit(x_train, y_train)
    elif algorithm == "LightGBM Classifier":
        model_path = os.path.join(MODEL_DIR, "lightgbm.pkl")
        model = joblib.load(model_path) if os.path.exists(model_path) else lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.01,
            max_depth=-1,
            num_leaves=31,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=42,
        ).fit(x_train, y_train)
    elif algorithm == "AdaBoost Classifier":
        model_path = os.path.join(MODEL_DIR, "adaboost.pkl")
        model = joblib.load(model_path) if os.path.exists(model_path) else AdaBoostClassifier(
            n_estimators=50, learning_rate=0.01, random_state=42
        ).fit(x_train, y_train)
    else:
        model_path = os.path.join(MODEL_DIR, "stacking_sgd_pac.pkl")
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            estimators = [
                ("sgd", SGDClassifier(loss="log_loss", penalty="l2", max_iter=1000, tol=1e-3, random_state=42)),
                ("pac", PassiveAggressiveClassifier(C=1.0, max_iter=1000, tol=1e-3, random_state=42)),
            ]
            model = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(),
                n_jobs=-1,
                passthrough=False,
            ).fit(x_train, y_train)

    joblib.dump(model, model_path)
    y_pred = model.predict(x_test)
    try:
        y_score = model.predict_proba(x_test)
    except Exception:
        y_score = None

    return session, calculate_metrics(algorithm, y_pred, y_test, y_score)


def train_xgboost(session):
    return _train_model(session, "XGBoost Classifier")


def train_lightgbm(session):
    return _train_model(session, "LightGBM Classifier")


def train_adaboost(session):
    return _train_model(session, "AdaBoost Classifier")


def train_stacking(session):
    return _train_model(session, "Stacking Classifier (SGD + PAC)")


def plot_model_performance():
    if classification_metrics_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(14, 7))
    df_melt = classification_metrics_df.melt(
        id_vars="Algorithm",
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score",
    )
    sns.barplot(x="Algorithm", y="Score", hue="Metric", data=df_melt, ax=ax)
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Model Performance Comparison")
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "model_performance_comparison.png"))
    return fig


def render_flash_graph(name, session):
    if name != "Model Performance Comparison" and session.get("df") is None:
        return None, "Please upload dataset first."

    if name == "Distribution: Scam vs Legit":
        fig, ax = plt.subplots(figsize=(7, 5))
        session["df"]["label"].value_counts().sort_index().plot(kind="bar", color=["#5B8DEF", "#FF6B6B"], ax=ax)
        ax.set_title("Distribution of Scam vs Legit Messages")
    elif name == "Platform vs Scam/Legit":
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.countplot(x="platform", hue="label", data=session["df"], palette="Set2", ax=ax)
        ax.tick_params(axis="x", rotation=30)
        ax.set_title("Platform vs Scam/Legit")
    elif name == "Sentiment vs Message Length":
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.scatterplot(
            x="sentiment_score",
            y="message_length",
            hue="label",
            data=session["df"],
            palette="Set1",
            ax=ax,
        )
        ax.set_title("Sentiment Score vs Message Length")
    elif name == "Promised Return by Label":
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.boxplot(x="label", y="promised_return_pct", data=session["df"], palette="pastel", ax=ax)
        ax.set_title("Promised Return Percentage by Label")
    elif name == "Feature Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10, 8))
        numeric_cols = ["promised_return_pct", "sentiment_score", "message_length", "hour", "dayofweek", "is_weekend"]
        corr_matrix = session["df"][numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
        ax.set_title("Correlation Heatmap of Features")
    else:
        fig = plot_model_performance()
        if fig is None:
            return None, "No model metrics available. Train at least one model."

    fig.tight_layout()
    return fig, f"Showing: {name}"


def show_flash_graph(name, session):
    if name not in FLASH_GRAPH_NAMES:
        return None, session, "Invalid graph selection."
    session["graph_index"] = FLASH_GRAPH_NAMES.index(name)
    fig, msg = render_flash_graph(name, session)
    return fig, session, msg


def previous_flash_graph(session):
    session["graph_index"] = (session.get("graph_index", 0) - 1) % len(FLASH_GRAPH_NAMES)
    name = FLASH_GRAPH_NAMES[session["graph_index"]]
    fig, msg = render_flash_graph(name, session)
    return gr.update(value=name), fig, session, msg


def next_flash_graph(session):
    session["graph_index"] = (session.get("graph_index", 0) + 1) % len(FLASH_GRAPH_NAMES)
    name = FLASH_GRAPH_NAMES[session["graph_index"]]
    fig, msg = render_flash_graph(name, session)
    return gr.update(value=name), fig, session, msg


def list_model_graphs(model_label):
    if model_label not in MODEL_TO_ALGO:
        return gr.update(choices=[], value=None), "Select a model."
    prefix = MODEL_TO_ALGO[model_label].replace(" ", "_")
    files = []
    if os.path.isdir(RESULTS_DIR):
        files = sorted([f for f in os.listdir(RESULTS_DIR) if f.startswith(prefix) and f.lower().endswith(".png")])
    return gr.update(choices=files, value=files[0] if files else None), (
        "Found graphs." if files else "No graphs yet. Train the model first."
    )


def open_model_graph(filename):
    if not filename:
        return None, "Select a graph."
    image_path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(image_path):
        return None, "Graph file not found."
    return image_path, f"Showing: {filename}"


def predict_testdata(test_file, session):
    ok, msg = require_role(session, {"USER", "ADMIN"})
    if not ok:
        return msg, None
    if test_file is None:
        return "Please upload test CSV.", None
    try:
        test_df = pd.read_csv(test_file)
        if not os.path.exists(W2V_PATH):
            return "Word2Vec model missing. Please preprocess dataset first.", None
        w2v_model = Word2Vec.load(W2V_PATH)
        X_testdata, _, logs = preprocess_data_word2vec(
            test_df, is_train=False, w2v_model=w2v_model, apply_smote=False
        )
        stack_path = os.path.join(MODEL_DIR, "stacking_sgd_pac.pkl")
        if not os.path.exists(stack_path):
            return "Stacking model not found. Train it first.", None
        stack_model = joblib.load(stack_path)
        preds = stack_model.predict(X_testdata)
        le_target = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
        test_df["predictions"] = le_target.inverse_transform(preds)
        return f"Prediction complete.\n{logs}", test_df
    except Exception as exc:
        return f"Error: {str(exc)}", None


GUIDE_TEXT = """
### Guide (Use This Order)
1. Login as **ADMIN**.
2. Click **Upload Dataset**.
3. Click **Preprocess Dataset**.
4. Click **Dataset Splitting**.
5. Train models (**XGBoost / LightGBM / AdaBoost / Stacking**).
6. Use **Flash Graphs** and model graph selectors.

If training/splitting is clicked too early, you'll get: **Read the Guide**.

For prediction:
1. Login as **USER**.
2. Upload test CSV in Prediction section.
3. Click **Predict**.
"""


with gr.Blocks(title="Bitcoin Scam Detection (Gradio)") as demo:
    session_state = gr.State(init_session())

    gr.Markdown("# Bitcoin Scam Detection")

    with gr.Column(visible=True) as auth_page:
        gr.Markdown("## Login / Signup")
        guide_btn = gr.Button("Show Guide")
        guide_markdown = gr.Markdown(GUIDE_TEXT, visible=False)

        with gr.Group():
            gr.Markdown("### Login")
            with gr.Row():
                role = gr.Radio(["ADMIN", "USER"], value="ADMIN", label="Role")
                username = gr.Textbox(label="Username")
                password = gr.Textbox(label="Password", type="password")
            login_btn = gr.Button("Login")

        with gr.Group():
            gr.Markdown("### Signup (USER only)")
            with gr.Row():
                signup_username = gr.Textbox(label="New Username")
                signup_password = gr.Textbox(label="New Password", type="password")
                signup_confirm = gr.Textbox(label="Confirm Password", type="password")
            signup_btn = gr.Button("Signup")
            signup_status = gr.Textbox(label="Signup Status", interactive=False)

        login_status = gr.Textbox(label="Session Status", interactive=False, value="Please login.")

    with gr.Column(visible=False) as operations_page:
        with gr.Row():
            ops_role_status = gr.Markdown("Not logged in")
            logout_btn = gr.Button("Logout")

        with gr.Tab("Admin Workflow"):
            dataset_file = gr.File(label="Upload Dataset CSV", file_types=[".csv"], type="filepath")
            upload_btn = gr.Button("Upload Dataset")
            preprocess_btn = gr.Button("Preprocess Dataset")
            split_btn = gr.Button("Dataset Splitting")
            dataset_status = gr.Textbox(label="Workflow Status", interactive=False)
            dataset_preview = gr.Dataframe(label="Dataset Preview", interactive=False)

            with gr.Row():
                train_xgb_btn = gr.Button("Train XGBoost")
                train_lgb_btn = gr.Button("Train LightGBM")
                train_ada_btn = gr.Button("Train AdaBoost")
                train_stack_btn = gr.Button("Train Stacking SGD_PAC")
            train_logs = gr.Textbox(label="Training Output", lines=14, interactive=False)

        with gr.Tab("Flash Graphs"):
            with gr.Row():
                flash_dd = gr.Dropdown(choices=FLASH_GRAPH_NAMES, value=FLASH_GRAPH_NAMES[0], label="Flash Graphs")
                prev_btn = gr.Button("<")
                next_btn = gr.Button(">")
            flash_plot = gr.Plot(label="Graph")
            flash_status = gr.Textbox(label="Graph Status", interactive=False)

        with gr.Tab("Model Graphs"):
            model_selector = gr.Dropdown(choices=list(MODEL_TO_ALGO.keys()), label="Model")
            refresh_model_graphs_btn = gr.Button("Refresh Model Graph List")
            model_graph_dd = gr.Dropdown(choices=[], label="Available Graphs")
            open_model_graph_btn = gr.Button("Open Selected Graph")
            model_graph_image = gr.Image(label="Model Graph")
            model_graph_status = gr.Textbox(label="Model Graph Status", interactive=False)

        with gr.Tab("Prediction"):
            test_file = gr.File(label="Upload Test CSV", file_types=[".csv"], type="filepath")
            predict_btn = gr.Button("Predict")
            predict_status = gr.Textbox(label="Prediction Status", lines=8, interactive=False)
            predict_table = gr.Dataframe(label="Predictions", interactive=False)

    guide_btn.click(show_guide_content, outputs=[guide_markdown])
    signup_btn.click(
        signup,
        inputs=[signup_username, signup_password, signup_confirm],
        outputs=[signup_status],
    )
    login_btn.click(
        handle_login,
        inputs=[role, username, password, session_state],
        outputs=[session_state, login_status, auth_page, operations_page, ops_role_status],
    )
    logout_btn.click(
        handle_logout,
        inputs=[session_state],
        outputs=[session_state, login_status, auth_page, operations_page, ops_role_status],
    )

    upload_btn.click(
        upload_dataset,
        inputs=[dataset_file, session_state],
        outputs=[session_state, dataset_status, dataset_preview],
    )
    preprocess_btn.click(
        preprocess_dataset,
        inputs=[session_state],
        outputs=[session_state, dataset_status],
    )
    split_btn.click(
        split_dataset,
        inputs=[session_state],
        outputs=[session_state, dataset_status],
    )

    train_xgb_btn.click(train_xgboost, inputs=[session_state], outputs=[session_state, train_logs])
    train_lgb_btn.click(train_lightgbm, inputs=[session_state], outputs=[session_state, train_logs])
    train_ada_btn.click(train_adaboost, inputs=[session_state], outputs=[session_state, train_logs])
    train_stack_btn.click(train_stacking, inputs=[session_state], outputs=[session_state, train_logs])

    flash_dd.change(
        show_flash_graph,
        inputs=[flash_dd, session_state],
        outputs=[flash_plot, session_state, flash_status],
    )
    prev_btn.click(
        previous_flash_graph,
        inputs=[session_state],
        outputs=[flash_dd, flash_plot, session_state, flash_status],
    )
    next_btn.click(
        next_flash_graph,
        inputs=[session_state],
        outputs=[flash_dd, flash_plot, session_state, flash_status],
    )

    refresh_model_graphs_btn.click(
        list_model_graphs,
        inputs=[model_selector],
        outputs=[model_graph_dd, model_graph_status],
    )
    open_model_graph_btn.click(
        open_model_graph,
        inputs=[model_graph_dd],
        outputs=[model_graph_image, model_graph_status],
    )

    predict_btn.click(
        predict_testdata,
        inputs=[test_file, session_state],
        outputs=[predict_status, predict_table],
    )


if __name__ == "__main__":
    demo.launch()
