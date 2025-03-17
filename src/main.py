import sys
import os
import time
from src.utils.database import db

def main():
    print("Starting Smart Security System")
    
    # Test database connection
    print("Testing database connection...")
    try:
        session = db.get_session()
        print("Database connection successful!")
        session.close()
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)
    
    # Keep the container running
    print("System is running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 