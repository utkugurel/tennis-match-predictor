import os
import sys

# Add the project root to sys.path so we can import app and tennis_features
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import app

if __name__ == "__main__":
    app.run()
