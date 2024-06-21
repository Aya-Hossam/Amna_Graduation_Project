import sqlite3
import bcrypt 
from datetime import datetime, timedelta

# Connect to database
def get_db_connection():
    return sqlite3.connect("accounts.db")

# Create Database if not exists
def create_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                      user_id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      user_email TEXT NOT NULL UNIQUE, 
                      user_password TEXT NOT NULL,
                      user_name TEXT NOT NULL,
                      confirmed_on NUMERIC,
                      is_confirmed INTEGER NOT NULL DEFAULT 0,
                      created_on NUMERIC
                      )''')


    cursor.execute('''CREATE TABLE IF NOT EXISTS admin (
                          admin_id INTEGER PRIMARY KEY AUTOINCREMENT,
                          admin_email TEXT NOT NULL UNIQUE,
                          admin_password TEXT NOT NULL,
                          admin_username TEXT NOT NULL UNIQUE,
                          is_confirmed INTEGER NOT NULL DEFAULT 1
                          )''')

    conn.commit()
    conn.close()

# Run the function to create the tables
create_db()


# Save and close database
def commit_and_close(conn):
    conn.commit()
    conn.close()
    print("Connection to the database is closed.")

# Fetch all users from the database
def get_all_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT user_name, user_email, user_id FROM users")
    rows = cursor.fetchall()
    conn.close()
    users = [{'user_name': row[0], 'user_email': row[1], 'user_id': row[2]} for row in rows]
    return users

# Add a new user
def sign_up_user(email, user_password, username):
    conn = get_db_connection()
    hashed_password = hash_password(user_password)
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users(user_email, user_password, user_name) VALUES (?, ?, ?)", (email, hashed_password, username))
        commit_and_close(conn)
        return True
    except sqlite3.IntegrityError as e:
        print(e)
        commit_and_close(conn)
        return False

def sign_up_admin(email, admin_password, username):
    conn = get_db_connection()
    cursor = conn.cursor()

    # Check the number of existing admins
    cursor.execute("SELECT COUNT(*) FROM admin")
    count = cursor.fetchone()[0]
    
    if count >= 2:
        print("Cannot add more than 2 admins.")
        conn.close()
        return False
    
    hashed_password = hash_password(admin_password)
    try:
        cursor.execute("INSERT INTO admin(admin_email, admin_password, admin_username) VALUES (?, ?, ?)", 
                       (email, hashed_password, username))
        commit_and_close(conn)
        print("Admin added successfully.")
        return True
    except sqlite3.IntegrityError as e:
        print(f"Integrity error: {e}")
        conn.close()
        return False
    except Exception as e:
        print(f"Error: {e}")
        conn.close()
        return False

# Delete a user by user_id
def delete_user(user_id):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        commit_and_close(conn)
        return True
    except sqlite3.IntegrityError:
        commit_and_close(conn)
        return False

# Delete a user by email
def delete_user_by_email(email):
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE user_email = ?", (email,))
        commit_and_close(conn)
        return True
    except sqlite3.IntegrityError:
        commit_and_close(conn)
        return False

# Log in and validate user
def authenticate_user(user_email, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_password FROM users WHERE user_email = ?', (user_email,))
    result = cursor.fetchone()
    conn.close()
    if result:
        hashed_password = result[0]
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password)
    return False


# Log in and validate admin
def authenticate_admin(admin_email, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT admin_password FROM admin WHERE admin_email = ?', (admin_email,))
    result = cursor.fetchone()
    conn.close()
    if result:
        hashed_password = result[0]
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password)
    return False



# Hash password
def hash_password(password):
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed_password

# Get user confirmation status function
def get_user_conf_status(email):
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT is_confirmed FROM users WHERE user_email = ?', (email,))
    result = cursor.fetchone()
    conn.close()
    if result:
        confirmation_status = result[0]
        return confirmation_status
    return False

# Get admin confirmation status function
def get_admin_conf_status(email):
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT is_confirmed FROM admin WHERE admin_email = ?', (email,))
    result = cursor.fetchone()
    conn.close()
    if result:
        confirmation_status = result[0]
        return confirmation_status
    return False

# Update confirmation status of user
def update_confirmation_status(email,conf_date):
  conn = get_db_connection()
  cursor = conn.cursor()

  try:
    cursor.execute("""
      UPDATE users SET is_confirmed = ?, confirmed_on = ? WHERE user_email = ?
    """, (1, conf_date, email))
    conn.commit()
    return True
  except Exception as e:
    return False
  finally:
    conn.close()
    
# Get account creation date
def set_creation_date(email,create_date):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""UPDATE users SET created_on = ? WHERE user_email = ?""", (create_date, email))
        conn.commit()
        return True
    
    except Exception as e:
        return False
    
    finally:
        conn.close()


# get unverified users 
def get_unverified_users():
    conn = get_db_connection()  
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE is_confirmed = ?", (0,))
    results = cursor.fetchall()
    conn.close()
    return results


# Delete unverified users after 24 hours
def delete_expired_unverified_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    current_time = datetime.utcnow()
    twenty_four_hours_ago = current_time - timedelta(hours=24)
    cursor.execute('DELETE FROM users WHERE is_confirmed = 0 AND created_on <= ?', (twenty_four_hours_ago,))
    conn.commit()
    conn.close()
    
################################################################

"""This Section Deals With The Reset Password Process"""


# reset or forget password function
def reset_password(user_email, new_password):

    conn = get_db_connection()
    cursor = conn.cursor()

    # Find the user by email
    cursor.execute("SELECT user_id FROM users WHERE user_email = ?", (user_email,))
    user_id = cursor.fetchone()

    # Check if user exists
    if not user_id:
        print("User not found.")
        conn.close()
        return False

    # Hash the new password
    hashed_password = hash_password(new_password)

    # Update the user's password
    cursor.execute("UPDATE users SET user_password = ? WHERE user_id = ?", (hashed_password, user_id[0]))

    commit_and_close(conn)

    print("Password reset successfully!") 


def check_user_email(email):
  """
  This function checks if a user with the provided email address exists in the database.

  Args:
      email: The email address to check.

  Returns:
      True if the email exists, False otherwise.
  """
  # Connect to the database
  conn = get_db_connection()
  cursor = conn.cursor()
  
  cursor.execute("SELECT COUNT(*) FROM users WHERE user_email = ?", (email,))
  count = cursor.fetchone()[0]  # Get the first element (count) from the result tuple
  
  commit_and_close(conn)
  
  return count > 0  # Check if count is greater than 0 (user exists)



