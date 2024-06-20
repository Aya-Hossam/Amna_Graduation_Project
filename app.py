from flask import Flask, render_template, request, jsonify, url_for, session, redirect
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import preProcess_survey as surv
import pickle
import kickbox
from threading import Thread
from datetime import timedelta, datetime
import requests
from flask_mail import Message, Mail
from helpers import *
import os
from auth_token import generate_token, confirm_token
import admin_user as users
from flask_apscheduler import APScheduler
from apscheduler.schedulers.background import BackgroundScheduler
import survey_dataset as survDB



app = Flask(__name__)
app.secret_key = 'sama'
app.security_password_salt = 'aya'

# Valid file formats for CT scan tool
app.config['ALLOWED_MIME_TYPES'] = {'png', 'jpeg', 'jpg'}

app.config['SESSION_ PERMANENT'] = False
app.config['SESSION_COOKIE_DURATION'] = timedelta(minutes=1)

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587  # Port for TLS encryption (typical)
app.config['MAIL_USE_TLS'] = True  # Enable TLS encryption
app.config['MAIL_USERNAME'] = 'amna.breast.cancer.platform@gmail.com'
app.config['MAIL_PASSWORD'] = 'mnno ptby bwvt ciur'

mail = Mail(app)

# Ensure the database is created
users.create_db()
survDB.create_db()

# Template context processor function
@app.context_processor
def inject_vars():
  return dict(logged_in=is_logged_in(), is_admin=is_admin())

#---------------------------------------------------------------------
"""This Section Deals with The Home Page"""

# Function to redirect the user to the index
@app.route('/')
def index():
    return render_template('home.html')


# Function to redirect the user to the index through the navigation bar
@app.route('/home', methods=['GET'])
def home_nav():
    return render_template('home.html')

#---------------------------------------------------------------------
"""This Section Deals with The Login Process"""

# Function to check if the user is logged in
def is_logged_in():
    if session.get('is_logged_in'):
        logged_in_status = True
    else:
        logged_in_status = False
    return logged_in_status

# Function to check if the account type is an admin
def is_admin():
    if is_logged_in():
        account_type = session['account_type']
        if account_type == 'admin':
            return True
        else:
            return False

# Logout function
@app.route('/logout')
def logout():
    # Forget any user_id
    session.clear()

    # Redirect user to login form
    return render_template('home.html')

# Login generalized API
@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        # Forget previous user
        session.clear()

        if request.method == 'POST': # User reached rotue via POST (as by submitting a form via POST)
            # Get account information from request
            data = request.get_json()

            email = data.get('login-Email')
            password = data.get('login-password')
            account_type = data.get('account_type')
                
            # Check the type of the user account
            if account_type == 'admin':
                authentication_status = users.authenticate_admin(email,password)
                confirm_status = users.get_admin_conf_status(email)
            else:
                authentication_status = users.authenticate_user(email,password)
                confirm_status = users.get_user_conf_status(email)

            if authentication_status:
                
                if not confirm_status:
                    return jsonify({'message': 'Please Verify Your Account','Content-Type': 'application/json', 'status': 0}) , 400
                
                session['account_type'] = account_type
                session['is_logged_in'] = True

                return jsonify({'message': 'Login Successful','Content-Type': 'application/json','status': 1}) , 200
            
            else:
                return jsonify({'message': 'Invalid Credentials','Content-Type': 'application/json', 'status': 0}), 400

        else:
            # User reached route via GET
            return render_template('login.html')
        
    except Exception as e:
        return jsonify({'message': 'An Error Has Occured','Content-Type': 'application/json', 'status': 0}), 400


#---------------------------------------------------------------------

"""This Section Deals With The Registration Process"""

# Automatically delete unverified users every 24 hours from database
def auto_deleteUnverified():
    # Initialize the scheduler
    scheduler = APScheduler(app=app)
    scheduler.init_app(app)
    scheduler.start()
    scheduler.add_job(id='automatic_accounts_clear', trigger='interval', func=users.delete_expired_unverified_users, seconds=86400) 
# auto_deleteUnverified()    


# Register generalized API
@app.route('/register', methods=['POST','GET'])
def register():
    
    try:
        #User came from POST request by submitting a form
        if request.method == 'POST':
            # Get account information from request
            data = request.get_json()
            user_name = data.get('user-name')
            user_email = data.get('user-email')
            user_password = data.get('user-password')
            confirm_password = data.get('confirm-password')
            
            # Validate user data
            if not user_name or not user_email or not user_password or not confirm_password:
                return jsonify({'message': 'All fields are required', 'Content-Type': 'application/json', 'status': 0}), 400
            
            # Check if passwords match
            if user_password != confirm_password:
                return jsonify({'message': 'Password do not match', 'Content-Type': 'application/json', 'status': 0}), 400
            
            # Check if email is valid API
            # client   = kickbox.Client('live_d162d8ddd2de198b51163c6e4033ff21a37bc03f4f76cb92916a157053106048')
            # kickboxObj  = client.kickbox()
            # response = kickboxObj.verify(user_email)
            # body = response.body
            # if body['result'] != 'deliverable':
            #     return jsonify({'message': 'Please enter a valid email', 'Content-Type': 'application/json', 'status': 0}), 400


            # Add user to the database
            add_user_status = users.sign_up_user(user_email, user_password, user_name)

            if add_user_status:
                
                # Store the registration date
                create_date = datetime.now()
                users.set_creation_date(user_email, create_date)
                
                # Generate token and send verification email
                token = generate_token(user_email, app.secret_key, app.security_password_salt)
                
                confirm_url = url_for('confirm_email', token=token, _external=True)
                html = render_template('activate_email_content.html', confirm_url=confirm_url)
                subject = "Please confirm your email"
                
                if send_email(user_email, subject, html):
                    return jsonify({'message': 'Registration complete! Please verify your account.', 'Content-Type': 'application/json', 'status': 1}), 200
                else:
                    delete_user_status = users.delete_user_by_email(user_email)
                    return jsonify({'message': 'An error occured while sending the validation link.', 'Content-Type': 'application/json', 'status': 0}), 400
                
            else:
                return jsonify({'message': 'User already exists', 'Content-Type': 'application/json','status': 0}), 401
            
        # User came to the route through GET
        else:
            return render_template('register.html')
    except Exception as e:
        print(e)
        return jsonify({'message': 'An Error Has Occured','Content-Type': 'application/json', 'status': 0}), 403

# Verify the email address
@app.route('/confirm/<token>')
def confirm_email(token):
    message = ''
    try:
        email = confirm_token(token, secret_key=app.secret_key, password_salt=app.security_password_salt)
    except:
        message = 'An unexpected error has occured.'
    
    # Confirmation link has expired (more than 1 hour has passed)
    if not email:
        message = 'The confirmation link is invalid or has expired.'
        return render_template('email_verification_status_page.html', message=message)
    
    conf_status = users.get_user_conf_status(email)
    if conf_status:
        message = 'Account already confirmed. Please login.'
    else:
        conf_date = datetime.now()
        users.update_confirmation_status(email,conf_date)
        message = 'You have confirmed your account. Thanks!'
    return render_template('email_verification_status_page.html', message=message)

# Function to send verification emails
def send_email(to, subject, template):
    try:
        msg = Message(
            subject,
            recipients=[to],
            html=template,
            sender=app.config['MAIL_USERNAME']
        )
        mail.send(msg)
        # email_thread = Thread(target=mail.send, args=(msg))
        # email_thread.start()
        return True
    except Exception as e:
        return False

# Function to get the verification page
@app.route('/verify_email')
def verify_page():
    return render_template('email_verification_status_page.html')

# Function to resend confirmation email
@app.route('/resend', methods=['POST'])
def resend_confirmation():
    try:
        data = request.get_json()
        user_email = data.get('user-email')
        
        if not user_email:
            return jsonify({'message': 'Please provide the email to which the verification email should be sent.', 'Content-Type': 'application/json', 'status': 0}), 400
            
        token = generate_token(user_email, app.secret_key, app.security_password_salt)
        confirm_url = url_for('confirm_email', token=token, _external=True)
        html = render_template('activate_email_content.html', confirm_url=confirm_url)
        subject = "Please confirm your email"
            
        if send_email(user_email, subject, html):
            return jsonify({'message': 'A new confirmation email has been sent.', 'Content-Type': 'application/json', 'status': 1}), 200
        else:
            return jsonify({'message': 'An error occured while sending the validation link.', 'Content-Type': 'application/json', 'status': 0}), 400  
        
    except Exception as e:
        print(e)
        return jsonify({'message': 'Oops! An error has occured. Please try again.', 'Content-Type': 'application/json', 'status': 0}), 400
        
        


#---------------------------------------------------------------------

"""This Section Deals With Creating A New Password"""

@app.route('/new_password', methods=['GET'])
def new_password_form():
    return render_template('new_password.html')

@app.route('/new_password', methods=['POST'])
def post_new_password():
    try:
        if request.method == "POST":
            
            data = request.get_json()
            email = data.get('user-email')
            
            user = users.check_user_email(email)
            
            if user:
                send_reset_password_email(email)
                return jsonify({'message': 'Reset email has been sent!', 'Content-Type': 'application/json', 'status': 1}), 200
            else:
                return jsonify({'message': "This email address doesn't exist.", 'Content-Type': 'application/json', 'status': 0}), 400
        
    except Exception as e:
        return jsonify({'message': 'Oops! An error has occured please try again.','Content-Type': 'application/json', 'status': 0}), 400

# Function to send reset link to email address
def send_reset_password_email(email):
    
    # Generate the unique reset password url
    reset_password_url = url_for('reset_password', 
                                 token=generate_token(email, app.secret_key, app.security_password_salt),
                                 user_email=email,
                                 _external= True)
    
    # Define the email content
    email_body = render_template('reset_password/reset_password_email_html_content.html', reset_password_url=reset_password_url)
    subject = "Reset Your Password"
    
    # Send the reset link to the provided email address
    send_email_status = send_email(email, subject, email_body)      

# Reset password process
@app.route('/new_password/<token>/<user_email>', methods=["POST", "GET"])
def reset_password(token, user_email):
    
    if request.method == "POST":
        
        if is_logged_in():
            return redirect(url_for("index"))
        
        
        user = confirm_token(token, secret_key=app.secret_key, password_salt=app.security_password_salt)
        if not user:
            return redirect(url_for('reset_error'))
        
        password = request.form['new-password']
        
        users.reset_password(user_email, password)
        
        return redirect(url_for('reset_success'))
            
    else:
        
        return render_template("reset_password/token_change_password.html")
           
# Route for successful password reset
@app.route('/reset_success')
def reset_success():
    return render_template("reset_password/reset_password_success.html")

# Route for unsuccessful password reset
@app.route('/reset_error')
def reset_error():
    return render_template("reset_password/reset_password_error.html")

        
   
#---------------------------------------------------------------------

"""This Section Deals With The Risk Assessment Tool"""

def load_pkl(file_path):
    """Load a pre-trained model with pickle extension"""
    with open(file_path, 'rb') as file:
        return pickle.load(file)

@app.route('/survey', methods=['GET'])
@login_required('/survey')
def get_survey():
    return render_template('inner-page-survey.html')
    

@app.route('/survey', methods=['POST'])
def post_survey():
    
    try:
        
        if request.method == 'POST': # User reached rotue via POST (as by submitting a form via POST)
            

            # Load the pre-trained model
            model = load_pkl('survey.pkl')
            
            try:
                
                # Get the data from the request body
                data = request.get_json()
                if not data:
                    return jsonify({'message': 'All features are required','Content-Type': 'application/json', 'status': 0}), 400 
                
                # PreProcess the data
                preProcessed_data = surv.process_survey(data)
                
                # Check if all the features are available
                if not preProcessed_data:
                    return jsonify({'message': 'All features are required','Content-Type': 'application/json', 'status': 0}), 400 
                
                prediction = model.predict([preProcessed_data])[0]
                
                # Store the data in the database
                database_labels = ["age_group_5_years", "race_eth", "first_degree_hx", "age_menarche", "age_first_birth", "BIRADS_breast_density", "menopaus", 
                                   "bmi_group", "biophx", "high_bmi_post_menopause", "family_hist_dense", "menarche_first_birth", "breast_cancer_history"]
                surveyDict = {}
                preProcessed_data.append(int(prediction))
                for i in range(len(preProcessed_data)):
                    surveyDict[database_labels[i]] = preProcessed_data[i]
                
                # survDB.add_survey_data(surveyDict)
                
                
                if prediction == 1:
                    prediction_label = "High Risk"
                elif prediction == 0:
                    prediction_label = "Low Risk"
                
                return jsonify({'message': f'Your prediction is: {prediction_label}','prediction': prediction_label, 'Content-Type': 'application/json', 'status': 1}), 200
                
            except Exception as e:
                print(e)
                return jsonify({'message': 'An Error Has Occured','Content-Type': 'application/json', 'status': 0}), 401
        
    except Exception as e:
        print(e)
        return jsonify({'message': 'An Error Has Occured','Content-Type': 'application/json', 'status': 0}), 403

#---------------------------------------------------------------------

"""This Section Deals With The CT Scan Analysis Tool"""

@app.route('/ct_scan', methods=['GET'])
@login_required('/ct_scan')
def get_ct():
    return render_template('ct_scan.html')


def load_model_fn(model_path):  # Renamed function to avoid conflicts
    model = None
    model = load_model(model_path)  # Load the pre-trained model
    if model:
        return model
    else:
        return False

def preprocess_image(image):
    # Preprocess the image
    image = image.resize((150, 150))
    image = image.convert('L')
    image = np.array(image)
    image = image.reshape((1, 150, 150, 1))
    image = image.astype('float32') / 255.0
    return image

# Function to validate the uploaded ct scan image file type
def allowed_mime_type(mimetype):
  """
  Checks if the MIME type is allowed for CT scans.
  """
  return mimetype in app.config['ALLOWED_MIME_TYPES']


@app.route('/ct_scan', methods=['POST'])
def post_ct():
    
    # Load the pre-trained model
    model = load_model_fn('cnn_model(trained).h5')
    
    if request.method == 'POST':
        try:
            
            if 'image' not in request.files:
                return jsonify({'message': 'Please upload a CT scan image.', 'Content-Type': 'application/json', 'status': 0}), 400
                
            image_file = request.files['image']
            
            # Get the file type of the image
            filename = image_file.filename
            file_type = filename.split('.')[-1].lower()
            print(file_type)

            
            if not allowed_mime_type(file_type):
                return jsonify({'message':'Please upload a CT scan image.','prediction': None, 'Content-Type': 'application/json', 'status': 0}), 400
            
            if not image_file or not (image_file.filename == '') or not (image_file.content_length == 0):
                try:
                    # Open the image file using PIL
                    img = Image.open(image_file, mode='r')
                    
                    # Preprocess the image
                    processed_image = preprocess_image(img)
                    
                    # Make prediction using the loaded model
                    prediction = model.predict(processed_image)
                    predicted_class_index = np.argmax(prediction)

                    # Map the predicted class index back to its original label
                    class_labels = ["benign", "malignant", "normal"]
                    predicted_class_label = class_labels[predicted_class_index]
                    
                    if predicted_class_label == "malignant":
                        return jsonify({'message': f'Your prediction is: {predicted_class_label}',
                                        'prediction': predicted_class_label,
                                        'Content-Type': 'application/json',
                                        'status': 1,
                                        'recommendation': 
                                            """It's recommended that you seek immediate medical attention!\n
                                            These are things you can do to decrease your risk:\n
                                            1. Maintain a healthy weight.\n
                                            2. Stay physically active. \n
                                            3. Limit alcohol intake. \n
                                            4. Eat a healthy diet rich in fruits, vegetables, and whole grains. \n
                                            5. Discuss hormone replacement therapy (HRT) with your doctor to understand risks and benefits. \n
                                            6. Talk to your doctor about genetic testing if you have a strong family history. \n
                                            7. Schedule regular mammograms starting at a recommended age (often 40 or 45).
                                            """}), 200
                    
                    else:
                        return jsonify({'message': f'Your prediction is: {predicted_class_label}','prediction': predicted_class_label, 'Content-Type': 'application/json', 'status': 1}), 200
                    
                    
                
                except Exception as e:
                    print(e)
                    return jsonify({'message':'An error has occured','prediction': None, 'Content-Type': 'application/json', 'status': 0}), 401
            else:
                # Handle the possibility that no image is provided.
                return jsonify({'message':'Please upload a CT scan image.','prediction': None, 'Content-Type': 'application/json', 'status': 0}), 400
                
        except Exception as e:
            print(e)
            return jsonify({'message':'An error has occured','prediction': None, 'Content-Type': 'application/json', 'status': 0}), 401

#---------------------------------------------------------------------

"""This Section Deals With The Articles"""

# Route to get the articles page
@app.route('/articles', methods=['GET'])
def articles():
  return render_template('articles.html')

@app.route('/age1', methods=['GET'])
def age_article1():
    return render_template('articles/age1.html')

@app.route('/first_degree2', methods=['GET'])
def first_degree_article2():
    return render_template('articles/first-degree2.html')

@app.route('/first_period3', methods=['GET'])
def first_period_article3():
    return render_template('articles/first-period3.html')

@app.route('/fist_birth4', methods=['GET'])
def first_birth_article4():
    return render_template('articles/first-birth4.html')

@app.route('/breast_density5', methods=['GET'])
def breast_density_article5():
    return render_template('articles/breast-density5.html')

@app.route('/menopause6', methods=['GET'])
def menopause_article6():
    return render_template('articles/menopause6.html')

@app.route('/bmi7', methods=['GET'])
def bmi_article7():
    return render_template('articles/bmi7.html')

#---------------------------------------------------------------------

"""This Section Deals With The Admin"""

# Function to add new admin account
def add_admin_account(email, password, username):
    users.sign_up_admin(email, password, username)

# Route to get the admin dashboard
@app.route('/admin', methods=['GET'])
def admin_dashboard():
    return render_template('admin.html')

# Route to fetch all users
@app.route('/admin/get_users', methods=['GET'])
def get_all_users_route():
    try:
        user = users.get_all_users()
        return jsonify(user)
    except Exception as e:
        return jsonify({'error': str(e)}) , 500
    
# Route to add user for admin
@app.route('/admin/add_user', methods=['POST'])
def add_user():

    data = request.get_json()
    user_name = data.get('user-name')
    user_email = data.get('user-email')
    user_password = data.get('user-password')
    confirm_password = data.get('confirm-password')
    
    # Validate user data
    if not user_name or not user_email or not user_password or not confirm_password:
        return jsonify({'message': 'All fields are required', 'Content-Type': 'application/json', 'status': 0}), 400
            
    # Check if passwords match
    if user_password != confirm_password:
        return jsonify({'message': 'Passwords do not match', 'Content-Type': 'application/json', 'status': 0}), 400
    
    # Check if email is valid API
    # client   = kickbox.Client('live_d162d8ddd2de198b51163c6e4033ff21a37bc03f4f76cb92916a157053106048')
    # kickboxObj  = client.kickbox()
    # response = kickboxObj.verify(user_email)
    # body = response.body
    # if body['result'] != 'deliverable':
    #     return jsonify({'message': 'Please enter a valid email', 'Content-Type': 'application/json', 'status': 0}), 400

    
    # Call the sign_up function to add the user
    success = users.sign_up_user(user_email, user_password, user_name)
    
    # Return appropriate response based on success or failure
    if success:
        create_date = datetime.now()
        users.set_creation_date(user_email, create_date)
        return jsonify({'message': 'User added successfully. Please verify your email!', 'Content-Type': 'application/json', 'status': 1}) , 201
    else:
        return jsonify({'message': 'Account already exists.', 'Content-Type': 'application/json', 'status': 0}) , 500

# Route to delete a user
@app.route('/admin/delete_user/<int:user_id>', methods=['DELETE'])
def delete_user_api(user_id):
    # Call the delete_user function to delete the user
    delete_successful = users.delete_user(user_id)
    
    # Return appropriate response based on success or failure
    if delete_successful:
        return jsonify({'message': f'User with ID {user_id} has been deleted successfully'}) , 200
    else:
        return jsonify({'error': f'Failed to delete user with ID {user_id}'}) , 500

#---------------------------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True, port=3000)
