"""Fail closed for production until authentication/ownership are implemented."""
import os


def validate_runtime():
    environment = os.getenv('APP_ENV', 'local').strip().lower()
    if environment not in {'local', 'test'}:
        raise RuntimeError(
            'Only local/test mode is supported. Production requires verified '
            'authentication, image/session ownership checks, and HTTPS deployment.')
    return environment
