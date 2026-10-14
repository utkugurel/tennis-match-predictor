import os
import sys

# Make the project root importable (app, tennis, tennis_features).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import server as app  # noqa: E402  Vercel serves the WSGI `app`
