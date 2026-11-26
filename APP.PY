
from flask import (
    Flask, render_template, request, url_for,
    send_from_directory, redirect, flash, send_file, session
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import (
    LoginManager, UserMixin, login_user,
    login_required, logout_user, current_user
)

from pathlib import Path
from contextlib import closing
from io import BytesIO

import os
import uuid
import json
import sqlite3
import secrets
import numpy as np


# ==========================================
# 1. FLASK CONFIGURATION
# ==========================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
SCALOGRAM_DIR = BASE_DIR / "static" / "generated"
MODEL_DIR = BASE_DIR / "model" / "training_results"

MODEL_FILE = MODEL_DIR / "resnet18_ecg_best.pth"
RESULTS_FILE = MODEL_DIR / "training_results.json"
REPORT_FILE = MODEL_DIR / "classification_report.txt"

DATABASE = BASE_DIR / "heartcare.db"
SECRET_FILE = BASE_DIR / ".secret_key"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SCALOGRAM_DIR.mkdir(parents=True, exist_ok=True)

if not SECRET_FILE.exists():
    SECRET_FILE.write_text(secrets.token_hex(32))

app.config["SECRET_KEY"] = SECRET_FILE.read_text().strip()

# Browser-session login configuration
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_REFRESH_EACH_REQUEST"] = False
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {"csv", "txt"}

model = None
class_names = None
torch = None


# ==========================================
# 2. LOGIN MANAGER
# ==========================================

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


# ==========================================
# 3. DATABASE
# ==========================================

def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db


def init_db():

    with closing(get_db()) as db:

        with db:

            db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL
                )
            """)

            db.execute("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    normal_percentage REAL NOT NULL,
                    abnormal_percentage REAL NOT NULL,
                    sampling_rate REAL NOT NULL,
                    samples INTEGER NOT NULL,
                    duration REAL NOT NULL,
                    scalogram_filename TEXT NOT NULL,
                    waveform_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)

            # Migrate existing database if necessary
            existing_columns = {
                row["name"]
                for row in db.execute(
                    "PRAGMA table_info(analysis_history)"
                ).fetchall()
            }

            if "waveform_json" not in existing_columns:

                db.execute("""
                    ALTER TABLE analysis_history
                    ADD COLUMN waveform_json TEXT
                """)

                print("Database migration: waveform_json added.")

            db.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_analysis_history_user_created
                ON analysis_history(user_id, created_at)
            """)

    print("Database initialized successfully.")


# ==========================================
# 4. USER MODEL
# ==========================================

class User(UserMixin):

    def __init__(self, user_id, username, email):
        self.id = str(user_id)
        self.username = username
        self.email = email


@login_manager.user_loader
def load_user(user_id):

    with closing(get_db()) as db:

        user = db.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()

    if user:
        return User(
            user["id"],
            user["username"],
            user["email"]
        )

    return None


init_db()


# ==========================================
# 5. FILE VALIDATION
# ==========================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ==========================================
# 6. LOAD RESNET18 MODEL
# ==========================================

def load_model():

    global model, class_names, torch

    if model is not None:
        return

    import torch as torch_module
    import torch.nn as nn
    from torchvision import models

    torch = torch_module
    torch.set_num_threads(1)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=False
    )

    class_names = checkpoint["class_names"]

    network = models.resnet18(weights=None)

    network.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(
            network.fc.in_features,
            len(class_names)
        )
    )

    network.load_state_dict(
        checkpoint["model_state_dict"]
    )

    network.eval()
    model = network

    del checkpoint

    print("ResNet18 loaded successfully.")
    print("Classes:", class_names)



# ==========================================
# 7. READ ECG FILE
# ==========================================

def read_ecg_file(filepath):

    import pandas as pd

    # First check whether the file has a header
    raw = pd.read_csv(
        filepath,
        sep=None,
        engine="python",
        comment="#",
        header=None
    )

    if raw.empty:
        raise ValueError("ECG file is empty.")

    # Detect whether the first row contains column names
    first_row = pd.to_numeric(
        raw.iloc[0],
        errors="coerce"
    )

    has_header = first_row.isna().any()

    if has_header:
        data = pd.read_csv(
            filepath,
            sep=None,
            engine="python",
            comment="#"
        )
    else:
        data = raw.copy()
        data.columns = [
            f"column_{i}" for i in range(len(data.columns))
        ]

    # Convert columns to numeric where possible
    time_column = None

    for column in data.columns:

        name = str(column).strip().lower()

        if name in {
            "time", "time_s", "time(sec)",
            "timestamp", "time_seconds"
        }:
            time_column = column
            break

    # Estimate sampling frequency from time column
    sampling_rate = 100.0

    if time_column is not None:

        times = pd.to_numeric(
            data[time_column],
            errors="coerce"
        ).dropna().to_numpy(dtype=np.float64)

        if len(times) > 1:

            intervals = np.diff(times)
            intervals = intervals[intervals > 0]

            if len(intervals) > 0:
                sampling_rate = 1.0 / float(
                    np.median(intervals)
                )

    # Find the ECG signal column
    preferred_names = {
        "ecg", "ecg_mv", "lead_i", "leadi",
        "lead_ii", "leadii", "signal",
        "voltage", "amplitude"
    }

    preferred_column = None

    for column in data.columns:

        if str(column).strip().lower() in preferred_names:
            preferred_column = column
            break

    if preferred_column is not None:

        signal_series = pd.to_numeric(
            data[preferred_column],
            errors="coerce"
        )

    else:

        # Exclude time column before selecting signal
        signal_data = data.drop(
            columns=[time_column]
            if time_column is not None else [],
            errors="ignore"
        )

        numeric_data = signal_data.apply(
            pd.to_numeric,
            errors="coerce"
        )

        numeric_data = numeric_data.dropna(
            axis=1,
            how="all"
        )

        if numeric_data.empty:
            raise ValueError(
                "No numeric ECG signal column found."
            )

        # Select the first available numeric signal column
        signal_series = numeric_data.iloc[:, 0]

    signal = signal_series.dropna().to_numpy(
        dtype=np.float32
    )

    if len(signal) < 64:
        raise ValueError(
            "At least 64 ECG samples are required."
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError("Invalid ECG signal values.")

    if sampling_rate <= 0:
        raise ValueError("Invalid sampling rate.")

    # Print information about selected input
    print("\n========== ECG FILE INFORMATION ==========")
    print("File:", filepath)
    print("Columns:", list(data.columns))
    print("Selected ECG column:", preferred_column)
    print("Sampling rate:", sampling_rate)
    print("Number of samples:", len(signal))
    print("First 15 values:", signal[:15])
    print("Minimum:", np.min(signal))
    print("Maximum:", np.max(signal))
    print("Mean:", np.mean(signal))
    print("Standard deviation:", np.std(signal))
    print("===========================================\n")

    return signal, float(sampling_rate)


# ==========================================
# 8. GENERATE CWT SCALOGRAM
# ==========================================

def generate_scalogram(signal, sampling_rate, output_path):

    import pywt
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    signal = np.asarray(signal, dtype=np.float32)

    # Normalize the current uploaded ECG signal
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)

    if signal_std < 1e-8:
        raise ValueError(
            "ECG signal has almost no variation."
        )

    normalized_signal = (
        signal - signal_mean
    ) / signal_std

    # Generate Continuous Wavelet Transform
    scales = np.arange(1, 65)

    coefficients, frequencies = pywt.cwt(
        normalized_signal,
        scales,
        "morl",
        sampling_period=1.0 / sampling_rate
    )

    scalogram = np.abs(coefficients)

    # Debug information
    print("\n========== CWT DATA CHECK ==========")
    print("CWT shape:", scalogram.shape)
    print("CWT minimum:", np.min(scalogram))
    print("CWT maximum:", np.max(scalogram))
    print("CWT mean:", np.mean(scalogram))
    print("CWT standard deviation:", np.std(scalogram))
    print("====================================\n")

    # Create a new image for this upload
    fig, ax = plt.subplots(figsize=(8, 5))

    image = ax.imshow(
        scalogram,
        extent=[
            0,
            len(normalized_signal) / sampling_rate,
            scales[-1],
            scales[0]
        ],
        aspect="auto",
        cmap="jet",
        interpolation="nearest"
    )

    ax.set_title("ECG Scalogram")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Wavelet Scale")

    fig.colorbar(
        image,
        ax=ax,
        label="Magnitude"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("New scalogram saved:", output_path)

    return output_path


# ==========================================
# 9. ECG PREDICTION
# ==========================================

def predict_ecg(image_path):

    load_model()

    from torchvision import transforms
    from PIL import Image

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    with Image.open(image_path) as image:

        image = image.convert("RGB")

        image_tensor = image_transform(
            image
        ).unsqueeze(0)

    with torch.inference_mode():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    if "Normal" not in class_names or "Abnormal" not in class_names:
        raise ValueError("Normal and Abnormal classes are required.")

    normal_index = class_names.index("Normal")
    abnormal_index = class_names.index("Abnormal")

    normal_percentage = (
        probabilities[normal_index].item() * 100
    )

    abnormal_percentage = (
        probabilities[abnormal_index].item() * 100
    )

    if normal_percentage > abnormal_percentage:

        predicted_class = "Normal"
        confidence = normal_percentage

    else:

        predicted_class = "Abnormal"
        confidence = abnormal_percentage

    del image_tensor, outputs, probabilities

    return (
        predicted_class,
        confidence,
        normal_percentage,
        abnormal_percentage
    )


# ==========================================
# 10. TRAINING RESULTS
# ==========================================

def get_training_results():

    results = {}
    report = ""

    if RESULTS_FILE.exists():

        with open(
            RESULTS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            results = json.load(file)

    if REPORT_FILE.exists():

        report = REPORT_FILE.read_text(
            encoding="utf-8"
        )

    return results, report


# ==========================================
# 11. REGISTRATION
# ==========================================

@app.route("/register", methods=["GET", "POST"])
def register():

    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if len(username) < 3:
            flash("Username must have at least 3 characters.")
            return redirect(url_for("register"))

        if len(username) > 40:
            flash("Username cannot exceed 40 characters.")
            return redirect(url_for("register"))

        if not email or len(email) > 120 or "@" not in email:
            flash("Enter a valid email address.")
            return redirect(url_for("register"))

        if len(password) < 8:
            flash("Password must have at least 8 characters.")
            return redirect(url_for("register"))

        password_hash = generate_password_hash(
            password,
            method="pbkdf2:sha256",
            salt_length=16
        )

        try:

            with closing(get_db()) as db:

                with db:

                    db.execute(
                        """
                        INSERT INTO users
                        (username, email, password_hash)
                        VALUES (?, ?, ?)
                        """,
                        (username, email, password_hash)
                    )

            flash("Account created successfully. Please login.")
            return redirect(url_for("login"))

        except sqlite3.IntegrityError:

            flash("This email is already registered. Please login.")
            return redirect(url_for("login"))

    return render_template("auth/register.html")


# ==========================================
# 12. LOGIN
# ==========================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        with closing(get_db()) as db:

            user = db.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,)
            ).fetchone()

        if user and check_password_hash(
            user["password_hash"], password
        ):

            session.clear()
            session.permanent = False

            login_user(
                User(
                    user["id"],
                    user["username"],
                    user["email"]
                ),
                remember=False
            )

            flash("Login successful.")
            return redirect(url_for("dashboard"))

        flash("Invalid email or password. Please try again.")

    return render_template("auth/login.html")


# ==========================================
# 13. LOGOUT
# ==========================================

@app.route("/logout", methods=["POST"])
@login_required
def logout():

    logout_user()
    session.clear()

    flash("You have successfully logged out.")

    return redirect(url_for("home"))


# ==========================================
# 14. HOME PAGE
# ==========================================

@app.route("/")
def home():

    return render_template("index.html")


# ==========================================
# 15. DASHBOARD AND HISTORY
# ==========================================

@app.route("/dashboard")
@login_required
def dashboard():

    with closing(get_db()) as db:

       history = db.execute(
    """
    SELECT *
    FROM analysis_history
    WHERE user_id = ?
    GROUP BY filename
    ORDER BY MAX(created_at) DESC
    LIMIT 10
    """,
    (current_user.id,)
).fetchall()

    normal_count = sum(
        1 for item in history
        if item["predicted_class"] == "Normal"
    )

    abnormal_count = len(history) - normal_count

    return render_template(
        "dashboard.html",
        history=history,
        total=len(history),
        normal_count=normal_count,
        abnormal_count=abnormal_count
    )



# ==========================================
# 16. ECG PREDICTION ROUTE
# ==========================================

@app.route("/predict", methods=["POST"])
@login_required
def predict():

    if "ecg_file" not in request.files:
        flash("Please select an ECG file.")
        return redirect(url_for("dashboard"))

    file = request.files["ecg_file"]

    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("dashboard"))

    if not allowed_file(file.filename):
        flash("Only CSV and TXT files are supported.")
        return redirect(url_for("dashboard"))

    filename = secure_filename(file.filename)

    # Unique ID for every upload
    unique_id = uuid.uuid4().hex[:10]

    saved_filename = f"{unique_id}_{filename}"
    upload_path = UPLOAD_DIR / saved_filename

    scalogram_filename = f"{unique_id}_scalogram.png"
    scalogram_path = SCALOGRAM_DIR / scalogram_filename

    try:

        # Save uploaded file
        file.save(upload_path)

        print("\n")
        print("########################################")
        print("          NEW ECG UPLOAD")
        print("########################################")
        print("Filename:", filename)
        print("Upload ID:", unique_id)

        # Read current ECG data
        signal, sampling_rate = read_ecg_file(
            upload_path
        )

        # Additional input debugging
        print("\n========== ECG DATA CHECK ==========")
        print("Uploaded file:", filename)
        print("Sampling rate:", sampling_rate)
        print("Number of samples:", len(signal))
        print("First 15 values:", signal[:15])
        print("Minimum:", np.min(signal))
        print("Maximum:", np.max(signal))
        print("Mean:", np.mean(signal))
        print("Standard deviation:", np.std(signal))
        print("====================================")

        # Generate scalogram for this exact input
        generate_scalogram(
            signal,
            sampling_rate,
            scalogram_path
        )

        # ResNet18 prediction
        (
            predicted_class,
            confidence,
            normal_percentage,
            abnormal_percentage
        ) = predict_ecg(scalogram_path)

        print("\n========== PREDICTION ==========")
        print("Class:", predicted_class)
        print("Confidence:", confidence)
        print("Normal:", normal_percentage)
        print("Abnormal:", abnormal_percentage)
        print("Scalogram:", scalogram_filename)
        print("================================\n")

        # Existing model training information
        results, report = get_training_results()

        # Prepare waveform
        if len(signal) > 1200:

            indexes = np.linspace(
                0,
                len(signal) - 1,
                1200,
                dtype=int
            )

            waveform = signal[indexes].tolist()

        else:

            waveform = signal.tolist()

        # Save analysis in database
        with closing(get_db()) as db:

            with db:

                cursor = db.execute(
                    """
                    INSERT INTO analysis_history (
                        user_id,
                        filename,
                        predicted_class,
                        confidence,
                        normal_percentage,
                        abnormal_percentage,
                        sampling_rate,
                        samples,
                        duration,
                        scalogram_filename,
                        waveform_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        current_user.id,
                        filename,
                        predicted_class,
                        round(confidence, 2),
                        round(normal_percentage, 2),
                        round(abnormal_percentage, 2),
                        round(sampling_rate, 2),
                        len(signal),
                        round(len(signal) / sampling_rate, 2),
                        scalogram_filename,
                        json.dumps(waveform)
                    )
                )

                analysis_id = cursor.lastrowid

        # Show result page
        return render_template(
            "result.html",

            analysis_id=analysis_id,
            filename=filename,

            predicted_class=predicted_class,
            confidence=round(confidence, 2),

            normal_percentage=round(
                normal_percentage, 2
            ),

            abnormal_percentage=round(
                abnormal_percentage, 2
            ),

            sampling_rate=round(sampling_rate, 2),
            samples=len(signal),

            duration=round(
                len(signal) / sampling_rate, 2
            ),

            waveform=waveform,

            scalogram=url_for(
                "static",
                filename=f"generated/{scalogram_filename}"
            ),

            results=results,
            report=report
        )

    except Exception as error:

        app.logger.exception(
            "ECG processing failed"
        )

        flash(f"ECG analysis failed: {error}")

        return redirect(url_for("dashboard"))


# ==========================================
# 17. VIEW SAVED ANALYSIS
# ==========================================

@app.route("/history/<int:analysis_id>")
@login_required
def history_detail(analysis_id):

    with closing(get_db()) as db:

        item = db.execute(
            """
            SELECT * FROM analysis_history
            WHERE id = ? AND user_id = ?
            """,
            (analysis_id, current_user.id)
        ).fetchone()

    if item is None:
        return "Analysis not found.", 404

    waveform = json.loads(
        item["waveform_json"] or "[]"
    )

    scalogram_url = url_for(
        "static",
        filename=f"generated/{item['scalogram_filename']}"
    )

    return render_template(
        "history_detail.html",
        item=item,
        waveform=waveform,
        scalogram_url=scalogram_url
    )



# ==========================================
# 18. COMPLETE PDF REPORT
# ==========================================

@app.route("/report/<int:analysis_id>.pdf")
@login_required
def download_report(analysis_id):

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import (
        getSampleStyleSheet,
        ParagraphStyle
    )
    from reportlab.lib.units import mm
    from reportlab.lib.enums import TA_CENTER
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image,
        PageBreak,
        Preformatted
    )
    from reportlab.graphics.shapes import Drawing, PolyLine
    from xml.sax.saxutils import escape

    # ------------------------------------------
    # FETCH ANALYSIS FROM DATABASE
    # ------------------------------------------

    with closing(get_db()) as db:

        item = db.execute(
            """
            SELECT * FROM analysis_history
            WHERE id = ? AND user_id = ?
            """,
            (analysis_id, current_user.id)
        ).fetchone()

    if item is None:
        return "Report not found.", 404

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=20 * mm,
        title="HeartCare AI ECG Analysis Report",
        author="HeartCare AI"
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="HCReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#10243B"),
        fontSize=22,
        leading=28
    ))

    styles.add(ParagraphStyle(
        name="HCSection",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#087F8C"),
        spaceBefore=12,
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name="HCSmall",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11
    ))

    story = []

    def safe(value):
        if value is None:
            return "Not available"
        return escape(str(value))

    def section(title):
        story.append(
            Paragraph(escape(title), styles["HCSection"])
        )

    def add_table(rows, widths=None):

        table = Table(
            rows,
            colWidths=widths,
            repeatRows=1
        )

        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#10243B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F1F5F9")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7)
        ]))

        story.append(table)

    def add_image(image_path, max_width=165 * mm,
                  max_height=100 * mm):

        image_path = Path(image_path)

        if not image_path.is_file():
            story.append(Paragraph(
                "Image file not available.",
                styles["HCSmall"]
            ))
            return

        try:
            img = Image(str(image_path))

            scale = min(
                max_width / img.imageWidth,
                max_height / img.imageHeight
            )

            img.drawWidth = img.imageWidth * scale
            img.drawHeight = img.imageHeight * scale

            story.append(img)

        except Exception:
            story.append(Paragraph(
                "Unable to include this image.",
                styles["HCSmall"]
            ))

    # ------------------------------------------
    # REPORT HEADER
    # ------------------------------------------

    story.append(Paragraph(
        "HeartCare AI",
        styles["HCReportTitle"]
    ))

    story.append(Paragraph(
        "COMPREHENSIVE ECG ANALYSIS REPORT",
        styles["Heading2"]
    ))

    story.append(Spacer(1, 5 * mm))

    story.append(Paragraph(
        f"Report ID: HC-{analysis_id}",
        styles["Normal"]
    ))

    story.append(Paragraph(
        f"Username: {safe(current_user.username)}",
        styles["Normal"]
    ))

    story.append(Paragraph(
        f"Email: {safe(current_user.email)}",
        styles["Normal"]
    ))

    story.append(Paragraph(
        f"Analysis Date: {safe(item['created_at'])}",
        styles["Normal"]
    ))

    story.append(Spacer(1, 8 * mm))

    # ------------------------------------------
    # 1. PREDICTION SUMMARY
    # ------------------------------------------

    section("1. ECG Prediction Summary")

    add_table([
        ["Parameter", "Result"],
        ["Uploaded ECG File", safe(item["filename"])],
        ["Predicted Class", safe(item["predicted_class"])],
        ["Model Confidence", f"{float(item['confidence']):.2f}%"],
        ["Normal Score", f"{float(item['normal_percentage']):.2f}%"],
        ["Abnormal Score", f"{float(item['abnormal_percentage']):.2f}%"]
    ], [65 * mm, 100 * mm])

    # ------------------------------------------
    # 2. SIGNAL INFORMATION
    # ------------------------------------------

    section("2. ECG Signal Information")

    add_table([
        ["Parameter", "Value"],
        ["Sampling Rate", f"{item['sampling_rate']} Hz"],
        ["Total Samples", str(item["samples"])],
        ["Signal Duration", f"{item['duration']} seconds"]
    ], [65 * mm, 100 * mm])

    # ------------------------------------------
    # 3. ECG WAVEFORM
    # ------------------------------------------

    section("3. Uploaded ECG Waveform")

    waveform = json.loads(item["waveform_json"] or "[]")

    if isinstance(waveform, list) and len(waveform) >= 2:

        try:
            values = [
                float(v)
                for v in waveform
                if v is not None and np.isfinite(float(v))
            ]

            if len(values) >= 2:

                max_points = 1200
                step = max(1, len(values) // max_points)
                plot_values = values[::step][:max_points]

                low = min(plot_values)
                high = max(plot_values)

                if high == low:
                    high = low + 1

                width = 165 * mm
                height = 48 * mm

                points = []

                for index, value in enumerate(plot_values):

                    x = (
                        index / max(1, len(plot_values) - 1)
                    ) * width

                    y = (
                        (value - low) / (high - low)
                    ) * (height - 8) + 4

                    points.extend([x, y])

                drawing = Drawing(width, height)

                drawing.add(PolyLine(
                    points,
                    strokeColor=colors.HexColor("#078B91"),
                    strokeWidth=0.8
                ))

                story.append(drawing)

                story.append(Paragraph(
                    f"Displayed waveform points: {len(plot_values)}. "
                    f"Stored samples: {len(values)}.",
                    styles["HCSmall"]
                ))

            else:
                story.append(Paragraph(
                    "Insufficient waveform data.",
                    styles["HCSmall"]
                ))

        except Exception:
            story.append(Paragraph(
                "Waveform could not be generated.",
                styles["HCSmall"]
            ))

    else:
        story.append(Paragraph(
            "Waveform data not available.",
            styles["HCSmall"]
        ))

    # ------------------------------------------
    # 4. CWT SCALOGRAM
    # ------------------------------------------

    section("4. Generated CWT Scalogram")

    scalogram_path = (
        SCALOGRAM_DIR / item["scalogram_filename"]
    )

    add_image(
        scalogram_path,
        max_width=160 * mm,
        max_height=100 * mm
    )

    story.append(Paragraph(
        "This image represents the time-frequency "
        "transformation generated from the uploaded ECG signal.",
        styles["HCSmall"]
    ))

    # ------------------------------------------
    # 5. MODEL TRAINING PERFORMANCE
    # ------------------------------------------

    story.append(PageBreak())

    section("5. Model Training Performance")

    training_results = {}

    if RESULTS_FILE.exists():

        try:
            training_results = json.loads(
                RESULTS_FILE.read_text(encoding="utf-8")
            )
        except Exception:
            training_results = {}

    if training_results:

        metric_rows = [["Metric", "Value"]]

        def flatten_metrics(data, prefix=""):

            if isinstance(data, dict):

                for key, value in data.items():

                    new_key = (
                        f"{prefix}.{key}" if prefix else str(key)
                    )

                    yield from flatten_metrics(value, new_key)

            elif isinstance(data, list):

                for index, value in enumerate(data):

                    new_key = f"{prefix}[{index}]"

                    yield from flatten_metrics(value, new_key)

            else:
                yield prefix, str(data)

        for key, value in flatten_metrics(training_results):

            metric_rows.append([
                safe(key),
                safe(value)
            ])

        add_table(
            metric_rows,
            [75 * mm, 90 * mm]
        )

    else:

        story.append(Paragraph(
            "Training metrics file was not found.",
            styles["HCSmall"]
        ))

    # ------------------------------------------
    # 6. TRAINING GRAPHS
    # ------------------------------------------

    section("6. Model Training Graphs")

    graph_files = [
        ("Accuracy Graph", "accuracy_graph.png"),
        ("Loss Graph", "loss_graph.png"),
        ("Confusion Matrix", "confusion_matrix.png")
    ]

    for graph_title, graph_filename in graph_files:

        story.append(Paragraph(
            escape(graph_title),
            styles["Heading3"]
        ))

        graph_path = MODEL_DIR / graph_filename

        add_image(
            graph_path,
            max_width=155 * mm,
            max_height=85 * mm
        )

        story.append(Spacer(1, 4 * mm))

    # ------------------------------------------
    # 7. CLASSIFICATION REPORT
    # ------------------------------------------

    story.append(PageBreak())

    section("7. Classification Report")

    if REPORT_FILE.exists():

        try:

            report_text = REPORT_FILE.read_text(
                encoding="utf-8"
            ).strip()

            if report_text:

                report_style = ParagraphStyle(
                    name="HCClassification",
                    fontName="Courier",
                    fontSize=7.5,
                    leading=10,
                    backColor=colors.HexColor("#F1F5F9"),
                    borderPadding=8
                )

                story.append(Preformatted(
                    report_text,
                    report_style
                ))

            else:

                story.append(Paragraph(
                    "Classification report is empty.",
                    styles["HCSmall"]
                ))

        except Exception:

            story.append(Paragraph(
                "Unable to read classification report.",
                styles["HCSmall"]
            ))

    else:

        story.append(Paragraph(
            "Classification report file was not found.",
            styles["HCSmall"]
        ))

    # ------------------------------------------
    # 8. DISCLAIMER
    # ------------------------------------------

    section("8. Medical Disclaimer")

    story.append(Paragraph(
        "HeartCare AI is an academic ECG analysis research prototype. "
        "The predicted class and model confidence are machine-learning "
        "outputs and are not a clinical diagnosis. Model confidence "
        "is not a calibrated probability of heart disease. "
        "Consult a qualified healthcare professional for medical "
        "interpretation.",
        styles["BodyText"]
    ))

    story.append(Spacer(1, 10 * mm))

    story.append(Paragraph(
        "End of HeartCare AI Report",
        styles["Italic"]
    ))

    # ------------------------------------------
    # PAGE NUMBER FOOTER
    # ------------------------------------------

    def add_page_number(canvas, doc):

        canvas.saveState()

        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#64748B"))

        canvas.drawString(
            16 * mm,
            10 * mm,
            "HeartCare AI | ECG Analysis Report"
        )

        canvas.drawRightString(
            A4[0] - 16 * mm,
            10 * mm,
            f"Page {doc.page}"
        )

        canvas.restoreState()

    document.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"HeartCare_Full_Report_{analysis_id}.pdf"
    )



# ==========================================
# 19. TRAINING GRAPH ROUTE
# ==========================================

@app.route("/training-assets/<path:filename>")
def training_assets(filename):

    return send_from_directory(
        MODEL_DIR,
        filename
    )


# ==========================================
# 20. RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5001)),
        debug=False,
        use_reloader=False
    )
