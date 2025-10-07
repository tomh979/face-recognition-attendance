import streamlit as st
import pandas as pd
import cv2
import numpy as np
from datetime import datetime, timedelta
import os
import json
from PIL import Image
import time
import glob

# Configure the page
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="📊",
    layout="wide"
)

class FaceRecognitionSystem:
    def __init__(self):
        self.setup_directories()
        self.init_session_state()
        
    def setup_directories(self):
        """Create necessary directories for the system"""
        os.makedirs("user_photos", exist_ok=True)
        os.makedirs("attendance_records", exist_ok=True)
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'users' not in st.session_state:
            st.session_state.users = self.load_users()
        if 'attendance_data' not in st.session_state:
            st.session_state.attendance_data = self.load_attendance()
        if 'recognized_user' not in st.session_state:
            st.session_state.recognized_user = None
        if 'recognition_confidence' not in st.session_state:
            st.session_state.recognition_confidence = 0
    
    def load_users(self):
        """Load users from JSON file - FIXED VERSION"""
        try:
            with open("users.json", "r") as f:
                users = json.load(f)
                
            # DEBUG: Show what we loaded
            st.sidebar.info(f"📁 Loaded {len(users)} users from users.json")
            
            # Check if users have photos, if not try to find them
            users_with_photos = 0
            for user in users:
                if 'created_at' not in user:
                    user['created_at'] = datetime.now().isoformat()
                
                # If user has no photos or empty photos array, try to find photos
                if not user.get('photos') or len(user.get('photos', [])) == 0:
                    # Look for photos in user_photos directory
                    photo_pattern = f"user_photos/{user['id']}_*"
                    matching_photos = glob.glob(photo_pattern)
                    if matching_photos:
                        user['photos'] = matching_photos
                        users_with_photos += 1
                        st.sidebar.success(f"✅ Found photos for {user['name']}")
                    else:
                        st.sidebar.warning(f"⚠️ No photos found for {user['name']}")
                else:
                    users_with_photos += 1
            
            st.sidebar.info(f"📸 {users_with_photos}/{len(users)} users have photos")
            return users
            
        except FileNotFoundError:
            st.sidebar.error("❌ users.json not found! Creating default structure...")
            # Create a proper users.json with your actual data
            default_users = [
                {
                    "id": 1,
                    "name": "Fatema salim ",
                    "photos": ["user_photos/5_Fatema salim _20251006_023036.jpg"],
                    "created_at": "2025-10-06T02:30:36.155658"
                },
                {
                    "id": 2,
                    "name": "Alzahra Ahmed", 
                    "photos": ["user_photos/6_Alzahra Ahmed_20251006_023112.jpg"],
                    "created_at": "2025-10-06T02:31:12.064554"
                },
                {
                    "id": 3,
                    "name": "Mahmoud ",
                    "photos": ["user_photos/7_Mahmoud _20251006_023127.jpg"],
                    "created_at": "2025-10-06T02:31:27.984265"
                },
                {
                    "id": 4,
                    "name": "Osama",
                    "photos": ["user_photos/8_Osama_20251006_023140.jpg"],
                    "created_at": "2025-10-06T02:31:40.717783"
                }
            ]
            
            # Save the default users
            with open("users.json", "w") as f:
                json.dump(default_users, f, indent=4)
            
            return default_users
    
    def save_users(self):
        """Save users to JSON file"""
        with open("users.json", "w") as f:
            json.dump(st.session_state.users, f, indent=4)
    
    def load_attendance(self):
        """Load attendance records"""
        try:
            with open("attendance.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_attendance(self):
        """Save attendance records"""
        with open("attendance.json", "w") as f:
            json.dump(st.session_state.attendance_data, f, indent=4)
    
    def add_user(self, name, photo_bytes):
        """Add a new user to the system"""
        new_id = max([user['id'] for user in st.session_state.users]) + 1 if st.session_state.users else 1
        
        # Save photo
        photo_filename = f"user_photos/{new_id}_{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        with open(photo_filename, "wb") as f:
            f.write(photo_bytes)
        
        # Add user to list
        new_user = {
            "id": new_id,
            "name": name,
            "photos": [photo_filename],
            "created_at": datetime.now().isoformat()
        }
        
        st.session_state.users.append(new_user)
        self.save_users()
        return new_id

    def extract_face_features(self, image_array):
        """Extract facial features with better face alignment"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            
            # Load face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces with more sensitive parameters for webcam photos
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.05,  # More sensitive scaling
                minNeighbors=6,    # More neighbor checks
                minSize=(50, 50),  # Larger minimum size for webcam
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) == 0:
                return None
                
            # Take the largest face (most prominent)
            faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
            x, y, w, h = faces[0]
            
            # Expand face region slightly to include more features
            padding = int(w * 0.1)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(gray.shape[1] - x, w + 2*padding)
            h = min(gray.shape[0] - y, h + 2*padding)
            
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to standard size for comparison
            face_roi = cv2.resize(face_roi, (150, 150))  # Larger size for better features
            
            # Apply histogram equalization for better contrast
            face_roi = cv2.equalizeHist(face_roi)
            
            # Multiple feature extraction methods
            hist = cv2.calcHist([face_roi], [0], None, [256], [0, 256])
            edges = cv2.Canny(face_roi, 50, 150)  # Adjusted thresholds
            
            # Additional feature: Local Binary Patterns (LBP)
            lbp = self.calculate_lbp(face_roi)
            
            # Normalize features
            hist = cv2.normalize(hist, hist).flatten()
            edges = cv2.normalize(edges, edges).flatten()
            lbp = cv2.normalize(lbp, lbp).flatten()
            
            return {
                'histogram': hist,
                'edges': edges,
                'lbp': lbp,
                'face_region': face_roi
            }
        except Exception as e:
            st.error(f"Feature extraction error: {e}")
            return None

    def calculate_lbp(self, image):
        """Calculate Local Binary Patterns for texture features"""
        try:
            # Simple LBP implementation
            height, width = image.shape
            lbp = np.zeros_like(image)
            
            for i in range(1, height-1):
                for j in range(1, width-1):
                    center = image[i, j]
                    code = 0
                    code |= (image[i-1, j-1] >= center) << 7
                    code |= (image[i-1, j] >= center) << 6
                    code |= (image[i-1, j+1] >= center) << 5
                    code |= (image[i, j+1] >= center) << 4
                    code |= (image[i+1, j+1] >= center) << 3
                    code |= (image[i+1, j] >= center) << 2
                    code |= (image[i+1, j-1] >= center) << 1
                    code |= (image[i, j-1] >= center) << 0
                    lbp[i, j] = code
            
            # Calculate LBP histogram
            lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            return lbp_hist
        except Exception as e:
            return np.zeros((256, 1))
    
    def compare_faces(self, features1, features2):
        """Compare two face feature sets with improved similarity calculation"""
        try:
            if features1 is None or features2 is None:
                return 0
                
            similarities = []
            
            # Compare histograms using multiple methods
            hist_correlation = cv2.compareHist(features1['histogram'], features2['histogram'], cv2.HISTCMP_CORREL)
            hist_intersect = cv2.compareHist(features1['histogram'], features2['histogram'], cv2.HISTCMP_INTERSECT)
            
            # Compare edges
            edges_similarity = cv2.matchTemplate(features1['edges'], features2['edges'], cv2.TM_CCOEFF_NORMED)[0][0]
            
            # Compare LBP features if available
            lbp_similarity = 0
            if 'lbp' in features1 and 'lbp' in features2:
                lbp_similarity = cv2.compareHist(features1['lbp'], features2['lbp'], cv2.HISTCMP_CORREL)
            
            # Normalize all similarities to 0-1 range
            hist_correlation = (hist_correlation + 1) / 2  # Convert from -1,1 to 0,1
            hist_intersect = min(hist_intersect / 1000000, 1.0)  # Normalize intersection
            edges_similarity = max(0, edges_similarity)  # Ensure non-negative
            lbp_similarity = (lbp_similarity + 1) / 2  # Convert from -1,1 to 0,1
            
            # Weighted combination (you can adjust these weights)
            total_similarity = (
                hist_correlation * 0.3 +
                hist_intersect * 0.2 +
                edges_similarity * 0.3 +
                lbp_similarity * 0.2
            )
            
            # Convert to percentage
            return max(0, total_similarity * 100)
        except Exception as e:
            return 0
    
    def recognize_face_from_camera(self, camera_image_array):
        """Recognize face by comparing with all stored user photos - IMPROVED VERSION"""
        try:
            # Extract features from camera image
            camera_features = self.extract_face_features(camera_image_array)
            
            if camera_features is None:
                return None, 0, "No face detected"
            
            best_match = None
            best_similarity = 0
            best_user = None
            
            # Compare with all users' photos
            for user in st.session_state.users:
                if user.get('photos') and len(user['photos']) > 0:
                    for photo_path in user['photos']:
                        try:
                            # Check if file exists
                            if not os.path.exists(photo_path):
                                continue
                                
                            # Load stored photo
                            stored_image = cv2.imread(photo_path)
                            if stored_image is None:
                                continue
                                
                            # Extract features from stored photo
                            stored_features = self.extract_face_features(stored_image)
                            
                            if stored_features is None:
                                continue
                            
                            # Compare features
                            similarity = self.compare_faces(camera_features, stored_features)
                            
                            # Update best match if better similarity - LOWERED THRESHOLD
                            if similarity > best_similarity and similarity > 50:  # Changed from 60 to 50
                                best_similarity = similarity
                                best_user = user
                                best_match = user['id']
                                
                        except Exception as e:
                            continue
            
            if best_user:
                return best_user['id'], best_similarity, best_user['name']
            else:
                return None, 0, "No match found"
                
        except Exception as e:
            return None, 0, f"Recognition error: {str(e)}"
    
    def detect_and_recognize_faces(self, image_array):
        """Detect faces and recognize them"""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            
            # Load face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Recognize the main face
            user_id, confidence, user_name = self.recognize_face_from_camera(image_array)
            
            # Draw rectangles and recognition info
            for (x, y, w, h) in faces:
                if user_id and confidence > 50:  # Changed from 60 to 50
                    # Green for recognized face
                    color = (0, 255, 0)
                    label = f"{user_name} ({confidence:.1f}%)"
                else:
                    # Red for unknown face
                    color = (0, 0, 255)
                    label = "Unknown Face"
                
                # Draw rectangle
                cv2.rectangle(image_array, (x, y), (x+w, y+h), color, 2)
                
                # Draw label background
                cv2.rectangle(image_array, (x, y-25), (x+w, y), color, -1)
                
                # Draw label text
                cv2.putText(image_array, label, (x, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return image_array, len(faces), user_id, confidence, user_name
            
        except Exception as e:
            st.error(f"Face detection error: {e}")
            return image_array, 0, None, 0, "Error"
    
    def record_attendance(self, user_id, checkin_type):
        """Record attendance for a user"""
        current_day = datetime.now().weekday()
        
        # Check if it's weekend (Friday=4, Saturday=5)
        if current_day in [4, 5]:
            return False, "Weekend - No attendance tracking"
        
        # Define check-in types and times
        checkin_types = {
            1: {"name": "Morning Start", "start": "08:00", "end": "10:00"},
            2: {"name": "Lunch Break Out", "start": "12:00", "end": "13:00"},
            3: {"name": "Lunch Break In", "start": "13:00", "end": "14:00"},
            4: {"name": "Evening End", "start": "16:25", "end": "19:00"}  # CHANGED TO 16:25
        }
        
        current_time = datetime.now().time()
        checkin_info = checkin_types.get(checkin_type)
        
        if not checkin_info:
            return False, "Invalid check-in type"
        
        # Check if within check-in time window
        start_time = datetime.strptime(checkin_info["start"], "%H:%M").time()
        end_time = datetime.strptime(checkin_info["end"], "%H:%M").time()
        
        if not (start_time <= current_time <= end_time):
            return False, f"Outside {checkin_info['name']} hours"
        
        # Check if already checked in today for this type
        today = datetime.now().strftime("%Y-%m-%d")
        existing_entry = next((entry for entry in st.session_state.attendance_data 
                             if entry['user_id'] == user_id 
                             and entry['date'] == today 
                             and entry['checkin_type'] == checkin_type), None)
        
        if existing_entry:
            return False, "Already checked in for this session"
        
        # Record attendance
        user_name = next((user['name'] for user in st.session_state.users if user['id'] == user_id), "Unknown")
        
        attendance_record = {
            'user_id': user_id,
            'user_name': user_name,
            'checkin_type': checkin_type,
            'checkin_name': checkin_info['name'],
            'timestamp': datetime.now().isoformat(),
            'date': today,
            'day': datetime.now().strftime("%A")
        }
        
        st.session_state.attendance_data.append(attendance_record)
        self.save_attendance()
        
        return True, f"Successfully checked in for {checkin_info['name']}"

def main():
    st.title("🎯 Face Recognition Attendance System")
    
    # Initialize system
    system = FaceRecognitionSystem()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose App Mode",
        ["Dashboard", "Live Camera Recognition", "Student Management", "Attendance Records", "System Info"]
    )
    
    if app_mode == "Dashboard":
        show_dashboard(system)
    elif app_mode == "Live Camera Recognition":
        show_live_camera_recognition(system)
    elif app_mode == "Student Management":
        show_user_management(system)
    elif app_mode == "Attendance Records":
        show_attendance_records(system)
    elif app_mode == "System Info":
        show_system_info(system)

def show_live_camera_recognition(system):
    st.header("📷 Face Recognition & Attendance")
    
    # TESTING MODE - Always allow attendance
    testing_mode = st.sidebar.checkbox("🔧 Testing Mode (Always allow attendance)", value=True)
    
    if testing_mode:
        current_checkin_type, current_checkin_name = 4, "Evening End (Testing)"
        st.info("🔧 **TESTING MODE ACTIVE** - Attendance recording enabled")
    else:
        # Weekend check
        current_day = datetime.now().weekday()
        if current_day in [4, 5]:
            st.error("🎉 **WEEKEND MODE** - No attendance tracking on Friday and Saturday!")
            return
        
        # Get current check-in period
        current_checkin_type, current_checkin_name = get_current_checkin_period(return_type=True)
        
        if not current_checkin_type:
            st.warning("⏰ Outside check-in hours - Camera will run but no attendance will be recorded")
        else:
            st.info(f"✅ **Current check-in period:** {current_checkin_name}")
    
    st.subheader("📸 Take Photo for Face Recognition")
    
    # Camera input
    camera_photo = st.camera_input("Look at the camera and click a photo for face recognition")
    
    if camera_photo is not None:
        # Convert the photo to OpenCV format
        image = Image.open(camera_photo)
        image_array = np.array(image)
        
        # Detect and recognize faces
        processed_image, face_count, user_id, confidence, user_name = system.detect_and_recognize_faces(image_array.copy())
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Original Photo", width='stretch')
        
        with col2:
            st.image(processed_image, caption=f"Face Recognition Result", width='stretch')
        
        # Show recognition results
        if face_count > 0:
            st.success(f"✅ Detected {face_count} face(s)!")
            
            if user_id and confidence > 50:
                st.success(f"👤 **Recognized:** {user_name}")
                st.info(f"🎯 **Confidence:** {confidence:.1f}%")
                
                # Store recognized user in session state
                st.session_state.recognized_user = user_id
                st.session_state.recognition_confidence = confidence
                
                if current_checkin_type or testing_mode:
                    # Auto-attendance for recognized user
                    if st.button(f"✅ Auto Check-in for {user_name}", type="primary", key="auto_checkin"):
                        checkin_type_to_use = current_checkin_type if current_checkin_type else 4
                        success, message = system.record_attendance(user_id, checkin_type_to_use)
                        
                        if success:
                            st.success(f"✅ {message}")
                            st.balloons()
                            checkin_name = current_checkin_name if current_checkin_name else "Evening End"
                            st.info(f"""
                            **Attendance Recorded Successfully!**
                            - **Student:** {user_name}
                            - **Check-in:** {checkin_name}
                            - **Time:** {datetime.now().strftime('%H:%M:%S')}
                            - **Date:** {datetime.now().strftime('%Y-%m-%d')}
                            - **Confidence:** {confidence:.1f}%
                            """)
                        else:
                            st.error(f"❌ {message}")
                else:
                    st.warning("⏰ Outside check-in hours - Attendance cannot be recorded")
            else:
                st.warning("❓ Face not recognized or low confidence. Please try again or use manual check-in.")
                
                # Manual check-in section
                st.write("---")
                st.subheader("Manual Check-in")
                
                selected_user_manual = st.selectbox(
                    "Select Student Manually",
                    [f"{user['id']} - {user['name']}" for user in st.session_state.users],
                    key="manual_attendance"
                )
                
                if st.button("📝 Manual Check-in", type="secondary"):
                    if selected_user_manual:
                        user_id = int(selected_user_manual.split(" - ")[0])
                        checkin_type_to_use = current_checkin_type if current_checkin_type else 4
                        success, message = system.record_attendance(user_id, checkin_type_to_use)
                        
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
        else:
            st.error("❌ No faces detected! Please ensure your face is clearly visible in the photo.")

def show_dashboard(system):
    st.header("📊 Attendance Dashboard")
    
    total_users = len(st.session_state.users)
    users_with_photos = len([user for user in st.session_state.users if user.get('photos') and len(user['photos']) > 0])
    total_checkins = len(st.session_state.attendance_data)
    current_day = datetime.now().weekday()
    is_weekend = current_day in [4, 5]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Students", total_users)
    with col2:
        st.metric("Students with Photos", f"{users_with_photos}/{total_users}")
    with col3:
        st.metric("Total Check-ins", total_checkins)
    with col4:
        day_status = "Weekend" if is_weekend else "Weekday"
        st.metric("Today", day_status)
    
    # Current check-in period
    current_checkin = get_current_checkin_period()
    st.metric("Current Period", current_checkin)
    
    if st.session_state.users:
        st.subheader("Student Photos")
        photos_cols = st.columns(4)
        photo_count = 0
        
        for user in st.session_state.users:
            if user.get('photos') and photo_count < 4:
                try:
                    photo_path = user['photos'][0]
                    if os.path.exists(photo_path):
                        with photos_cols[photo_count]:
                            st.image(photo_path, width=150, caption=user['name'])
                            photo_count += 1
                except:
                    continue
    
    st.subheader("Recent Activity")
    if st.session_state.attendance_data:
        recent_data = st.session_state.attendance_data[-5:][::-1]
        for record in recent_data:
            st.write(f"✅ **{record['user_name']}** - {record['checkin_name']} at {record['timestamp'][11:16]}")
    else:
        st.info("No recent activity recorded")

def show_user_management(system):
    st.header("👥 Student Management")
    
    st.subheader("Add New Student")
    
    with st.form("add_user_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_name = st.text_input("Full Name", placeholder="Enter student's full name")
        
        with col2:
            new_photo = st.file_uploader(
                "Upload Face Photo",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear face photo for recognition",
                key="new_user_photo"
            )
        
        submitted = st.form_submit_button("Add Student", type="primary")
        
        if submitted:
            if new_name and new_photo:
                user_id = system.add_user(new_name, new_photo.getvalue())
                st.success(f"✅ Student '{new_name}' added successfully with ID: {user_id}")
                st.rerun()
            else:
                st.error("❌ Please provide both name and photo")
    
    st.subheader("Registered Students")
    
    if not st.session_state.users:
        st.info("No students registered yet. Add students using the form above.")
    else:
        for user in st.session_state.users:
            with st.expander(f"👤 {user['name']} (ID: {user['id']})"):
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    if user.get('photos') and len(user['photos']) > 0:
                        try:
                            photo_path = user['photos'][0]
                            if os.path.exists(photo_path):
                                image = Image.open(photo_path)
                                st.image(image, width=150, caption="Student Photo")
                            else:
                                st.error("❌ Photo file not found")
                        except Exception as e:
                            st.error(f"❌ Error loading photo: {e}")
                    else:
                        st.info("📷 No photo uploaded")
                
                with col2:
                    st.write(f"**Student ID:** {user['id']}")
                    st.write(f"**Photos:** {len(user.get('photos', []))}")
                    
                    registered_date = user.get('created_at')
                    if registered_date:
                        try:
                            display_date = datetime.fromisoformat(registered_date).strftime('%Y-%m-%d')
                            st.write(f"**Registered:** {display_date}")
                        except:
                            st.write("**Registered:** Date unavailable")
                    
                    user_attendance = len([a for a in st.session_state.attendance_data 
                                         if a['user_id'] == user['id']])
                    st.write(f"**Total Check-ins:** {user_attendance}")
                
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{user['id']}", type="secondary"):
                        st.session_state.users = [u for u in st.session_state.users if u['id'] != user['id']]
                        system.save_users()
                        st.success(f"Student {user['name']} has been deleted.")
                        st.rerun()

def show_attendance_records(system):
    st.header("📈 Attendance Records")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_user = st.selectbox(
            "Filter by Student",
            ["All Students"] + [user['name'] for user in st.session_state.users]
        )
    
    with col2:
        date_filter = st.selectbox(
            "Date Range",
            ["Last 7 days", "Last 30 days", "All time"]
        )
    
    if st.button("Export Data"):
        if st.session_state.attendance_data:
            df = pd.DataFrame(st.session_state.attendance_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"attendance_export_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    if st.session_state.attendance_data:
        filtered_data = st.session_state.attendance_data.copy()
        
        if selected_user != "All Students":
            filtered_data = [record for record in filtered_data if record['user_name'] == selected_user]
        
        if date_filter == "Last 7 days":
            week_ago = datetime.now() - timedelta(days=7)
            filtered_data = [record for record in filtered_data 
                           if datetime.fromisoformat(record['timestamp']).date() >= week_ago.date()]
        elif date_filter == "Last 30 days":
            month_ago = datetime.now() - timedelta(days=30)
            filtered_data = [record for record in filtered_data 
                           if datetime.fromisoformat(record['timestamp']).date() >= month_ago.date()]
        
        if filtered_data:
            display_data = []
            for record in filtered_data:
                display_data.append({
                    'Student': record['user_name'],
                    'Check-in Type': record['checkin_name'],
                    'Date': record['date'],
                    'Time': datetime.fromisoformat(record['timestamp']).strftime('%H:%M:%S'),
                    'Day': record['day']
                })
            
            df = pd.DataFrame(display_data)
            st.dataframe(df, width='stretch')
            
            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(filtered_data))
            with col2:
                unique_users = len(set([r['user_name'] for r in filtered_data]))
                st.metric("Unique Students", unique_users)
            with col3:
                unique_dates = len(set([r['date'] for r in filtered_data]))
                st.metric("Days Covered", unique_dates)
        else:
            st.info("No records match the selected filters")
    else:
        st.info("No attendance records available yet")

def show_system_info(system):
    st.header("🔧 System Information")
    
    st.subheader("User Database Status")
    st.write(f"**Total Users:** {len(st.session_state.users)}")
    
    users_with_photos = [user for user in st.session_state.users if user.get('photos') and len(user['photos']) > 0]
    st.write(f"**Users with Photos:** {len(users_with_photos)}")
    
    st.subheader("File System Check")
    st.write(f"**users.json exists:** {os.path.exists('users.json')}")
    st.write(f"**attendance.json exists:** {os.path.exists('attendance.json')}")
    st.write(f"**user_photos directory exists:** {os.path.exists('user_photos')}")
    
    if st.button("🔄 Reload User Data"):
        st.session_state.users = system.load_users()
        st.rerun()

def get_current_checkin_period(return_type=False):
    """Get the current check-in period"""
    current_time = datetime.now().time()
    current_day = datetime.now().weekday()
    
    if current_day in [4, 5]:
        return "Weekend" if not return_type else (None, "Weekend")
    
    checkin_periods = [
        (1, "Morning Start", "08:00", "10:00"),
        (2, "Lunch Break Out", "12:00", "13:00"),
        (3, "Lunch Break In", "13:00", "14:00"),
        (4, "Evening End", "16:25", "19:00")  # CHANGED TO 16:25
    ]
    
    for period_type, period_name, start_str, end_str in checkin_periods:
        start_time = datetime.strptime(start_str, "%H:%M").time()
        end_time = datetime.strptime(end_str, "%H:%M").time()
        
        if start_time <= current_time <= end_time:
            return period_name if not return_type else (period_type, period_name)
    
    return "Outside hours" if not return_type else (None, "Outside hours")

if __name__ == "__main__":
    main()