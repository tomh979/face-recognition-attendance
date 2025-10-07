# face-recognition-attendance
Face Recognition Attendance System with Streamlit and OpenCV
# 🎯 Face Recognition Attendance System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive face recognition-based attendance system built with Streamlit and OpenCV. Automatically track student attendance using facial recognition technology.

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Support](#-support)

</div>

## ✨ Features

### 🎯 Core Functionality
- **Real-time Face Recognition** - Advanced facial feature extraction and comparison
- **Automated Attendance Tracking** - Records attendance with timestamps
- **Multiple Check-in Periods** - Configurable time windows for different sessions
- **Student Management** - Add, view, and manage student profiles with photos
- **Attendance Reports** - Filter and export attendance records

### 🎨 User Interface
- **Streamlit Web Interface** - Modern, responsive web application
- **Live Camera Integration** - Real-time face capture and recognition
- **Dashboard Analytics** - Visual overview of attendance data
- **Manual Check-in Option** - Fallback for recognition failures

### ⚙️ Technical Features
- **Multi-feature Recognition** - Combines histogram, edge detection, and LBP features
- **Configurable Confidence** - Adjustable recognition thresholds (50% default)
- **Data Export** - CSV export functionality
- **Session Management** - Persistent data storage with JSON files

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Webcam for face recognition
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/tomh979/face-recognition-attendance.git
   cd face-recognition-attendance
Install dependencies

bash
pip install -r requirements.txt
Run the application

bash
streamlit run main.py
Open your browser and go to http://localhost:8501

📁 Project Structure
text
face-recognition-attendance/
├── main.py                 # Main application file
├── users.json              # Student database
├── attendance.json         # Attendance records
├── user_photos/           # Student photo directory
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
└── .gitignore            # Git ignore rules
🎮 Usage Guide
1. Dashboard
View overall attendance statistics

See recent activity

Check current check-in period status

View student photos

2. Adding Students
Navigate to Student Management

Click "Add New Student"

Enter student name and upload clear face photo

System automatically processes the photo for recognition

3. Taking Attendance
Go to Live Camera Recognition

Ensure good lighting and face visibility

Click photo using the camera interface

System automatically recognizes and records attendance

Use manual check-in if recognition fails

4. Viewing Reports
Access Attendance Records

Filter by student or date range

Export data as CSV if needed

⚙️ Configuration
Check-in Periods
The system supports four check-in periods:

Period	Time Window	Purpose
Morning Start	08:00 - 10:00	Start of day
Lunch Break Out	12:00 - 13:00	Lunch start
Lunch Break In	13:00 - 14:00	Lunch end
Evening End	16:25 - 19:00	End of day
Testing Mode
Enable Testing Mode in the sidebar to bypass time restrictions and always allow attendance recording.

🛠️ Technical Details
Face Recognition Algorithm
Face Detection: OpenCV Haar Cascades

Feature Extraction:

Histogram analysis

Edge detection (Canny)

Local Binary Patterns (LBP)

Similarity Calculation: Weighted combination of multiple features

Confidence Threshold: 50% minimum for recognition

🔧 Troubleshooting
Common Issues
"No faces detected"

Ensure good lighting

Face should be clearly visible

Remove obstructions (glasses, hats)

Low recognition confidence

Use high-quality reference photos

Ensure similar lighting conditions

Retake reference photos if needed

Camera not working

Check camera permissions

Ensure no other app is using camera

Test with system camera app

Performance Tips
Use clear, well-lit photos for student registration

Ensure consistent lighting during recognition

Close other camera-using applications

📊 System Requirements
Minimum
Python 3.8+

2GB RAM

Webcam

100MB storage

Recommended
Python 3.9+

4GB RAM

HD Webcam (720p+)

500MB storage

🛡️ Privacy & Security
All face data processed locally

No external API calls for recognition

Student photos stored securely

Compliance with privacy regulations

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
OpenCV for face detection capabilities

Streamlit for the web framework

Contributors and testers
