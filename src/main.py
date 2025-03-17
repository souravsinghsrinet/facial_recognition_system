import sys
import os
from PyQt5.QtWidgets import QApplication
from src.services.gui import MainWindow
from src.utils.database import db

def main():
    print("Starting Smart Security System")
    
    try:
        # Get configuration from environment variables or use defaults
        camera_id = int(os.getenv('CAMERA_ID', '0'))
        confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.6'))
        
        # Create the application
        app = QApplication(sys.argv)
        
        # Create and show the main window
        window = MainWindow(
            camera_id=camera_id,
            confidence_threshold=confidence_threshold
        )
        window.show()
        
        # Start the event loop
        return app.exec_()
    except Exception as e:
        print(f"Error starting application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())