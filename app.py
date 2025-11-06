import os
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
# --- NEW: Import security functions for hashing ---
from werkzeug.security import generate_password_hash, check_password_hash 
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)

# ---------------- Flask App Configuration ----------------
app = Flask(__name__)
# IMPORTANT: In a real app, use environment variables for this.
app.secret_key = 'a_very_secure_secret_key_for_brain_app' 

# Context processor to make datetime available in templates
@app.context_processor
def inject_global_vars():
    return dict(datetime=datetime)

# Upload configuration
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- PREPROCESSING CONFIGURATION (Based on your training) ---
NORMALIZATION_METHOD = '0-1' 
IMG_SIZE = (128, 128)
# -------------------------------------------------------------------

# ---------------- Database Configuration ----------------
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///brain_tumor_app.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ---------------- Model Loading ----------------
model = None
try:
    model = load_model('brain_tumor_model.keras', compile=False)
    if model:
        input_shape = model.input_shape
        logging.info(f"Deep learning model loaded successfully. Expected input shape: {input_shape}.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

# ---------------- Label Map ----------------
# NOTE: Verify this order matches your training class indices!
CLASS_NAMES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# ---------------- Database Models ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    # --- CHANGED: Password now stores the hashed version ---
    password = db.Column(db.String(256), nullable=False) 

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_email = db.Column(db.String(100), db.ForeignKey('user.email'), nullable=False)
    image_filename = db.Column(db.String(100), nullable=False)
    prediction_result = db.Column(db.String(100), nullable=False) 
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Initialize database tables
with app.app_context():
    # NOTE: You MUST delete the old brain_tumor_app.db file 
    # for the new 'User' model with the longer password field to take effect!
    db.create_all()
    logging.info("Database tables created/checked.")

# ---------------- Utility Functions ----------------
def allowed_file(filename):
    """Checks if the uploaded file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- Routes ----------------
@app.route('/')
def home():
    """Renders the landing page."""
    return render_template('home.html')

# --- FIX: Registration now uses HASHING ---
@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles user registration."""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        raw_password = request.form['password']

        if User.query.filter_by(email=email).first():
            flash('Email already registered. Please try logging in.', 'danger')
            return redirect(url_for('register'))

        # --- NEW: Hash the password before storing ---
        hashed_password = generate_password_hash(raw_password, method='pbkdf2:sha256') 
        
        user = User(username=username, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful. Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# --- FIX: Login now uses HASH CHECKING ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login."""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        # 1. Check if user exists
        if user:
            # 2. Check the provided password against the stored hash
            if check_password_hash(user.password, password):
                session['email'] = user.email
                flash(f'Welcome back, {user.username}!', 'success')
                return redirect(url_for('predict'))
            else:
                flash('Invalid email or password.', 'danger') # Generic message for security
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logs the user out."""
    session.pop('email', None)
    flash('Logged out successfully. See you next time!', 'info')
    return redirect(url_for('home'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Handles image upload, preprocessing, and model prediction."""
    if 'email' not in session:
        flash('Please login to access the prediction service.', 'warning')
        return redirect(url_for('login'))

    global model
    if model is None:
        flash('Prediction service is unavailable. Model failed to load.', 'danger')
        return redirect(url_for('home')) 

    if request.method == 'POST':
        file = request.files.get('image')
        filepath = None

        if not file or file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Invalid file type. Please upload a JPG, JPEG, or PNG image.', 'danger')
            return redirect(request.url)

        try:
            filename = secure_filename(file.filename)
            # Create a unique filename to prevent clashes and caching issues
            unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            # --- Preprocessing (Verified against your stated configuration) ---
            # Model Input: (128, 128, 3) and 0-1 normalization
            
            # Load the image with the correct target size and color mode
            img = image.load_img(filepath, target_size=IMG_SIZE, color_mode='rgb')
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Normalization: 0-1 (Dividing by 255.0)
            img_array = img_array / 255.0
            logging.info("Using 0-1 normalization.")
            
            # ----------------- Prediction -----------------
            prediction_array = model.predict(img_array)
            logging.info(f'Raw model output: {prediction_array[0]}')

            predicted_class_index = np.argmax(prediction_array[0])
            result = CLASS_NAMES[predicted_class_index]
            confidence = float(prediction_array[0][predicted_class_index] * 100)
            full_result_text = f"{result} ({confidence:.2f}%)"

            # ----------------- Save Prediction to DB -----------------
            new_prediction = Prediction(
                user_email=session['email'],
                image_filename=unique_filename, # Save the unique name
                prediction_result=full_result_text
            )
            db.session.add(new_prediction)
            db.session.commit()

            img_url = url_for('static', filename=f'uploads/{unique_filename}')
            return render_template('result.html',
                                   result=result,
                                   confidence=confidence,
                                   img_path=img_url)

        except Exception as e:
            logging.error(f'Prediction processing failed: {e}', exc_info=True)
            flash(f'An error occurred during prediction. Error: {e}', 'danger')
            # Clean up uploaded file if processing failed
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
            return redirect(request.url)

    # Pass model info to the template for user context
    model_input_shape = model.input_shape[1:3] if model else IMG_SIZE
    return render_template('predict.html', model_input_shape=model_input_shape)

@app.route('/dashboard')
def dashboard():
    """Displays the user's past prediction history."""
    if 'email' not in session:
        flash('Please login to view your dashboard.', 'warning')
        return redirect(url_for('login'))

    user_email = session['email']
    # Fetch predictions sorted by newest first
    predictions = Prediction.query.filter_by(user_email=user_email).order_by(Prediction.timestamp.desc()).all()

    return render_template('dashboard.html', predictions=predictions)


# ---------------- Run App ----------------
if __name__ == '__main__':
    # Set host to 0.0.0.0 for containerized environments
    app.run(debug=True, host='0.0.0.0')