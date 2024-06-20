from itsdangerous import URLSafeTimedSerializer


def generate_token(email,secret_key, password_salt):
    serializer = URLSafeTimedSerializer(secret_key)
    return serializer.dumps(email, salt=password_salt)


def confirm_token(token,secret_key, password_salt, expiration=3600):
    serializer = URLSafeTimedSerializer(secret_key)
    try:
        email = serializer.loads(
            token, salt=password_salt, max_age=expiration
        )
        return email
    except Exception:
        return False
