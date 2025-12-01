from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .conn import Base
from sqlalchemy import Column, Integer, Float, Text, ForeignKey, DateTime

from datetime import datetime
from sqlalchemy.orm import relationship



class Chart(Base):
    __tablename__ = "chart"

    id = Column(Integer, primary_key=True)
    uid = Column(Integer, nullable=False, )
    userid = Column(Text, nullable=False, )
    type = Column(Integer, nullable=True)   # Enum 타입으로 변경
    chart_type = Column(Text, nullable=True)   # Enum 타입으로 변경
    query_count = Column(Integer, nullable=False, default=0)

class TimeLog(Base):
    __tablename__ = "time_log"
    
    id = Column(Integer, primary_key=True)
    chart_id = Column(Integer, ForeignKey("chart.id"))
    step_name = Column(Text, nullable=False)  # "차트분별", "이미지처리", "전체시간" 등
    elapsed_time = Column(Float, nullable=True)  # 초 단위
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    
    chart = relationship("Chart", backref="time_logs")

class Question(Base):
    __tablename__ = "question"
    id = Column(Integer, primary_key=True)

    chart_id = Column(Integer, ForeignKey("chart.id"))
    
    content = Column(Text, nullable=False)
    answer_content = Column(Text, nullable=True)

    create_date = Column(DateTime, nullable=True, default=datetime.now)
    
    
    
    chart = relationship("Chart", backref="questions")