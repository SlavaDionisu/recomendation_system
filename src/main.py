import sys
import os

# Add the project root to the path so we can import from recommendation_system
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the FastAPI app from the recommendation system
from recommendation_system.recomendation_system_with_vectors import app

# If this script is run directly, start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)