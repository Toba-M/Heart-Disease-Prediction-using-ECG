from flask import Flask, request, jsonify, render_template,redirect,url_for,session,flash
from werkzeug.security import generate_password_hash,check_password_hash
import os,csv
import openai
import pandas as pd
import warnings
import os
import re
import cv2
import numpy as np
import pandas as pd
import shutil
import joblib
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
from skimage import img_as_float
import scipy.ndimage
import skimage.io
import skimage.color
import skimage.transform
import skimage.util
from tensorflow.keras.models import load_model
from dotenv import load_dotenv, find_dotenv
warnings.filterwarnings('ignore')
from chatbot import full_response

from clinical_ecg_functions import (
    build_ecg_feature_extractor_sequential,
    extract_time_domain_features,
    extract_morphological_features, 
    extract_frequency_features,
    enhance_contrast,
    normalize_image,
    extract_valid_ecg_contours,
    map_contours_to_signal,
    baseline_correction_ecg,
    extract_nonlinear_features
)
def extract_ecg_signal_from_image(image_path, num_points=500):
    img_read = skimage.io.imread(image_path)[..., :3]
    img = skimage.color.rgb2gray(img_read)
    img = skimage.img_as_float(img)
    #imgn = random_noise(img, var=0.1**2)
    img_bayes = denoise_wavelet(img, method='BayesShrink', wavelet='bior4.4', mode='soft', wavelet_levels=3, rescale_sigma=True)
    img_hybrid = denoise_wavelet(img_bayes, method='VisuShrink', wavelet='db4', mode='hard', wavelet_levels=3, rescale_sigma=True)
    normalized = normalize_image(img_hybrid)
    enhanced = enhance_contrast(normalized)
    thresholded = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 109, 74)
    inverted = cv2.bitwise_not(thresholded)
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    blurred = cv2.convertScaleAbs(blurred)
    blurred = cv2.bilateralFilter(blurred, 9, 50, 50)
    edges = cv2.Canny(blurred, 50, 150)
    contours =  extract_valid_ecg_contours(edges, min_area=0.5, min_width_fraction=0.005, debug=True)
    x, y = map_contours_to_signal(contours, edges.shape, num_points=num_points,interp_kind='linear')
    x = np.array(x,dtype=np.float64)
    y = np.array(y,dtype=np.float64)
    corrected_signal = baseline_correction_ecg(y, kernel_size=81)
    return x, corrected_signal

le = joblib.load(r"C:\Users\PC\Desktop\PROJECT2\clinical_label_encoder.pkl")
model = load_model(r"C:\Users\PC\Desktop\PROJECT2\clinical_mlp_model.keras")
rfc = joblib.load(r"C:\Users\PC\Desktop\PROJECT2\clinical_random_forest_model.pkl")
xgb = joblib.load(r"C:\Users\PC\Desktop\PROJECT2\clinical_xgboost_model.pkl")
scaler = joblib.load(r"C:\Users\PC\Desktop\PROJECT2\clinical_scaler.pkl")
imputer = joblib.load(r"C:\Users\PC\Desktop\PROJECT2\clinical_imputer.pkl")

LEAD_BOXES = {
    'I':   (26,   0, 477,  75),
    'II':  (26,  76, 477,  76),
    'III': (26, 155, 477,  87),
    'aVR': (26, 244, 477,  88),
    'aVL': (26, 334, 477,  75),
    'aVF': (26, 404, 477,  91),
    'V1':  (520,   0, 484,  81),
    'V2':  (520,  81, 484,  84),
    'V3':  (519, 165, 485,  77),
    'V4':  (519, 242, 485,  84),
    'V5':  (519, 326, 485,  84),
    'V6':  (519, 410, 485,  91),
}

PREDICTION_ROOT_DIR = "prediction_images"
UNKNOWN_DIR = os.path.join(PREDICTION_ROOT_DIR, "unknown")


def predict_ecg_condition(image_path):
    # Cleanup and setup
    if os.path.exists(UNKNOWN_DIR):
        shutil.rmtree(UNKNOWN_DIR)
    os.makedirs(UNKNOWN_DIR, exist_ok=True)

    # Load and split image
    img = cv2.imread(image_path)
    if img is None:
        return "⚠️ Could not load image."

    resized = cv2.resize(img, (1004, 580))
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_folder = os.path.join(UNKNOWN_DIR, base_name)
    os.makedirs(out_folder, exist_ok=True)

    for lead, (x, y, w, h) in LEAD_BOXES.items():
        crop = resized[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(out_folder, f"{lead}.png"), crop)

    ecg_feature_extractor = build_ecg_feature_extractor_sequential(input_shape=(500, 1), feature_dim=256)
    all_rows = []

    for lead_img in os.listdir(out_folder):
        lead_path = os.path.join(out_folder, lead_img)

        try:
            x_vals, sig = extract_ecg_signal_from_image(lead_path)
            sig = np.asarray(sig, dtype=np.float64).flatten()
        except Exception as e:
            print(f"⚠️ Signal extraction failed on {lead_path}: {e}")
            continue

        try:
            inp = sig.reshape(1, -1, 1).astype(np.float32)
            dl_vec = ecg_feature_extractor.predict(inp, verbose=0)[0]
        except Exception as e:
            print(f"❌ DL extractor failed on {lead_path}: {e}")
            dl_vec = np.zeros(256)

        try:
            td_dict = extract_time_domain_features(sig)
        except:
            td_dict = {}

        try:
            morph = extract_morphological_features(sig, duration=5, sample_count=len(sig))
        except:
            morph = {}

        try:
            freq_vec = extract_frequency_features(sig, fs=500)
        except:
            freq_vec = np.zeros(6)

        try:
            nl = extract_nonlinear_features(sig, fs=500)
        except:
            nl = {}

        all_rows.append({
            'image_id': base_name,
            'lead': lead_img,
            'deep_learning_features': dl_vec.tolist(),
            'time_domain_features': td_dict.tolist(),
            'morphological_features': morph.tolist(),
            'frequency_domain_features': freq_vec.tolist(),
            'non_linear_features': nl.tolist()
        })

    df = pd.DataFrame(all_rows)

    def concatenate_features(group):
        combined = {}
        feature_columns = ['deep_learning_features', 'time_domain_features',
                           'morphological_features', 'frequency_domain_features',
                           'non_linear_features']
        for col in feature_columns:
            combined[col] = [item for sublist in group[col] for item in sublist]
        return pd.Series(combined)

    combined_df = df.groupby('image_id').apply(concatenate_features).reset_index()
    combined_df['image_id'] = combined_df['image_id'].apply(lambda x: re.sub(r'[^A-Za-z]', '', str(x)))

    dl = np.vstack(combined_df['deep_learning_features'].values)
    td = np.vstack(combined_df['time_domain_features'].values)
    mf = np.vstack(combined_df['morphological_features'].values)
    fd = np.vstack(combined_df['frequency_domain_features'].values)
    nl = np.vstack(combined_df['non_linear_features'].values)
    X = np.hstack([dl, td, mf, fd, nl])

    scaled = scaler.transform(X)
    X_imputed = imputer.transform(scaled)

    pred = model.predict(X_imputed)
    predicted_class = np.argmax(pred, axis=1)
    confidence_scores = np.max(pred, axis=1)
    label = le.inverse_transform(predicted_class)

    return f"✅ Predicted Label: {label[0]}, Confidence: {confidence_scores[0]:.2f}"

# ─── Load env & OpenAI ──────────────────────
dotenv_path = find_dotenv()
if dotenv_path:
    load_dotenv(dotenv_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

# ─── Instantiate LLM, Tools, Agent, Memory ──
# (move your existing LangChain setup & full_response() here or import it)

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = "super_secret_key"
USER_FILE = "users.csv"
app.config['UPLOAD_FOLDER'] = 'static/files'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

if not os.path.exists(USER_FILE) or os.stat(USER_FILE).st_size == 0:
    with open(USER_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'email', 'password'])

# ────── Helper Functions ────── #
def get_users():
    users = {}
    if os.path.exists(USER_FILE):
        with open(USER_FILE, newline='') as f:
            reader = csv.reader(f)
            lines = list(reader)
            if not lines:
                print("⚠️ users.csv is empty")
                return users  

            # Skip header if present
            has_header = lines[0][0].lower() == 'name'
            rows = lines[1:] if has_header else lines

            for row in rows:
                if len(row) >= 3:
                    name, email, password = row[0], row[1], row[2]
                    users[email] = {'name': name, 'password': password}
    return users




def save_user(name, email, password):
    with open(USER_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, email, generate_password_hash(password)])



@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.get_json(force=True)
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({'reply': "I didn't catch that. Could you say it again?"}), 400

    # get the LLM response dict
    result = full_response(user_msg)
    return jsonify({
        'reply': result.get('ai_reply', "Sorry, something went wrong.")
    })

@app.route('/')
def index():
    return render_template('pject.html')

@app.route('/check-login')
def check_login():
    return jsonify({'logged_in': 'user' in session})

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm = request.form.get('cpassword')

        if password != confirm:
            return "⚠️ Passwords do not match"

        users = get_users()
        if email in users:
            return "⚠️ Email already exists"

        save_user(name, email, password)
        return jsonify({'success': True, 'message': "✅ Signup successful!", 'redirect': url_for('index') + "#login"})

    return render_template('signup.html')


    
@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    users = get_users()

    if email in users and check_password_hash(users[email]['password'], password):
        session['user'] = email
        session['name'] = users[email]['name']
        return jsonify({'success': True, 'redirect': url_for('predict')})
    else:
        return jsonify({'success': False, 'message': "⚠️ Invalid credentials. Please check your email and password."})


@app.route('/logout')
def logout():
    session.clear()
    flash("You’ve been logged out.", "info")
    return redirect(url_for('index'))

@app.route('/predict')
def predict():
    if 'user' not in session:
        flash("Login required to access this page.", "warning")
        return redirect(url_for('index'))  # or url_for('login') if separate login page
    return render_template('prediction.html', username=session.get('name'))


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = f"{file.filename}"
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return jsonify({'message': 'File uploaded successfully', 'filename': filename}), 200

@app.route('/predictfile', methods=['POST'])
def predictfile():
    data = request.get_json()
    filename = data.get('filename')

    if not filename:
        return jsonify({'error': 'No filename provided'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    result = predict_ecg_condition(file_path)

    if "Predicted Label" in result:
        parts = result.split("Confidence:")
        label = parts[0].split(":")[-1].strip().strip(',')
        confidence = parts[1].strip()
        return jsonify({'prediction': label, 'confidence': confidence})
    else:
        return jsonify({'error': result}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8000,debug=True)