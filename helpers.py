from flask import redirect, session, url_for
from functools import wraps


def login_required(route):  # Decorator accepts the route
  def decorator(f):
    """Decorator function to check if the user is logged in"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        
        # Check login status
        login_status = session.get('is_logged_in')
        
        if not login_status:  # Assuming you have a user object
            return redirect(url_for('login', next=route))  # Redirect with next param
        
        return f(*args, **kwargs)
    
    return decorated_function

  return decorator


def logout_required(func):
    """Decorator function to check if the user is logged out"""
    @wraps(func)
    def decorated_view(*args, **kwargs):
        login_status = session.get('is_logged_in')
        if login_status:
            return redirect(url_for('home_nav'))  # Redirect to homepage if logged in
        return func(*args, **kwargs)
    return decorated_view
