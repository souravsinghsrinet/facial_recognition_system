# Facial Recognition System

A comprehensive facial recognition solution for secure access control and user management. This system captures, analyzes, and logs facial data in real-time, providing a modern web interface for monitoring and administration.

## Project Overview

The Facial Recognition System is designed to:

- Detect and identify faces from webcam feeds or uploaded images
- Register new users with their facial data
- Track access attempts with timestamps and confidence scores
- Provide real-time notifications for access events
- Offer a user-friendly web interface for system management

## Tech Stack

### Backend
- **Python 3.8+**: Core programming language
- **Flask/SocketIO**: Web server and real-time communication
- **OpenCV**: Computer vision and image processing
- **face_recognition**: Facial detection and recognition library
- **PostgreSQL**: Database for user data and access logs
- **SQLAlchemy**: ORM for database interactions
- **Alembic**: Database migrations

### Frontend
- **Bootstrap 5**: Responsive UI framework
- **JavaScript/jQuery**: Client-side interactivity
- **HTML5/CSS3**: Web interface structure and styling
- **Font Awesome**: Icon library

### Infrastructure
- **Docker**: Containerization for easy deployment
- **docker-compose**: Multi-container orchestration

## Installation

### Prerequisites
- Docker and docker-compose installed
- Web camera (for live recognition)
- Git

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/facial_recognition_system.git
cd facial_recognition_system
```

2. Configure environment variables (optional):
   Create a `.env` file in the project root with the following variables:
```
POSTGRES_DB=facialrecognition
POSTGRES_USER=facialuser
POSTGRES_PASSWORD=yourpassword
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
CAMERA_ID=0
CONFIDENCE_THRESHOLD=0.5
```

3. Build and start the containers:
```bash
docker-compose up -d
```

## Usage

### Running the System

Start all services:
```bash
docker-compose up -d
```

Stop all services:
```bash
docker-compose down
```

View logs:
```bash
docker-compose logs -f
```

### Accessing the Web Interface

Open your browser and navigate to:
```
http://localhost:8080
```

### Key Features

1. **Dashboard**: View real-time camera feed and access events
2. **User Management**: Register and manage users
3. **Access Logs**: Review historical access attempts
4. **User Registration**: Add new users via webcam or image upload

### User Registration Methods

- **Webcam Capture**: Use your device's camera to capture a face image
- **Image Upload**: Upload an existing image file containing a face

## Development

### Project Structure

```
facial_recognition_system/
├── docker/               # Docker configuration files
│   ├── database/         # Database service Dockerfile
│   │   ├── Dockerfile    # Database service Dockerfile
│   │   └── init.sql      # Database initialization script
│   ├── face_recognition/ # Face recognition service Dockerfile
│   └── migrations/       # Database migrations Dockerfile
├── migrations/           # Alembic migration scripts
├── src/                  # Source code
│   ├── services/         # Core services
│   │   ├── face_recognition.py # Face recognition module
│   │   └── gui.py        # GUI module (for desktop app)
│   ├── utils/            # Utilities
│   │   └── database.py   # Database operations
│   └── web/              # Web interface
│       ├── server.py     # Flask web server
│       ├── static/       # Static assets (CSS, JS)
│       └── templates/    # HTML templates
└── docker-compose.yml    # Service orchestration
```

## Troubleshooting

- **Camera Issues**: If the webcam isn't detected, try setting `USE_UPLOAD_MODE=true` in your environment variables
- **Recognition Quality**: For better recognition, ensure good lighting and a clear face image
- **Database Connection**: Check PostgreSQL connection parameters if database errors occur

## License

This project is licensed under the MIT License - see the LICENSE file for details.
