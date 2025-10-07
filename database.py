import sqlalchemy as db
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import numpy as np
import json

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    user_id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    face_embedding = Column(Text)  # Serialized face embedding
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship with attendance records
    attendance_records = relationship("Attendance", back_populates="user")

class Attendance(Base):
    __tablename__ = 'attendance'
    
    record_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.user_id'))
    date = Column(String(10), nullable=False)  # YYYY-MM-DD format
    quarter_id = Column(String(10), nullable=False)  # YYYY-Q1, YYYY-Q2, etc.
    checkin1_time = Column(DateTime, nullable=True)  # Morning Start
    checkin2_time = Column(DateTime, nullable=True)  # Lunch Break Out
    checkin3_time = Column(DateTime, nullable=True)  # Lunch Break In
    checkin4_time = Column(DateTime, nullable=True)  # Evening End
    
    # Relationship with user
    user = relationship("User", back_populates="attendance_records")

class DatabaseManager:
    def __init__(self, database_url="sqlite:///attendance_system.db"):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def add_user(self, name, face_embedding):
        """Add a new user with face embedding"""
        user = User(name=name, face_embedding=json.dumps(face_embedding.tolist()))
        self.session.add(user)
        self.session.commit()
        return user.user_id
    
    def get_user_by_id(self, user_id):
        """Retrieve user by ID"""
        return self.session.query(User).filter(User.user_id == user_id).first()
    
    def get_all_users(self):
        """Retrieve all users"""
        return self.session.query(User).all()
    
    def record_attendance(self, user_id, checkin_type, timestamp=None):
        """Record attendance for a specific check-in type"""
        if timestamp is None:
            timestamp = datetime.datetime.now()
        
        date_str = timestamp.strftime("%Y-%m-%d")
        quarter_id = self._get_quarter_id(timestamp)
        
        # Check if record exists for today
        attendance = self.session.query(Attendance).filter(
            Attendance.user_id == user_id,
            Attendance.date == date_str
        ).first()
        
        if not attendance:
            attendance = Attendance(
                user_id=user_id,
                date=date_str,
                quarter_id=quarter_id
            )
            self.session.add(attendance)
        
        # Update the appropriate check-in time
        if checkin_type == 1:
            attendance.checkin1_time = timestamp
        elif checkin_type == 2:
            attendance.checkin2_time = timestamp
        elif checkin_type == 3:
            attendance.checkin3_time = timestamp
        elif checkin_type == 4:
            attendance.checkin4_time = timestamp
        
        self.session.commit()
        return attendance
    
    def _get_quarter_id(self, timestamp):
        """Calculate quarter ID from timestamp"""
        year = timestamp.year
        month = timestamp.month
        quarter = (month - 1) // 3 + 1
        return f"{year}-Q{quarter}"
    
    def get_attendance_records(self, start_date=None, end_date=None, user_id=None):
        """Get attendance records with optional filters"""
        query = self.session.query(Attendance)
        
        if user_id:
            query = query.filter(Attendance.user_id == user_id)
        if start_date:
            query = query.filter(Attendance.date >= start_date)
        if end_date:
            query = query.filter(Attendance.date <= end_date)
        
        return query.all()
    
    def get_compliance_rate(self, quarter_id=None, user_id=None):
        """Calculate compliance rate for batch or individual"""
        # Implementation for compliance calculation
        pass
    
    def close(self):
        """Close database session"""
        self.session.close()