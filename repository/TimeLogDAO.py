from sqlalchemy.orm import Session
from database.schema import TimeLog
from datetime import datetime

class TimeLogDAO:
    @staticmethod
    def create(db: Session, time_log_data: dict) -> TimeLog:
        time_log = TimeLog(**time_log_data)
        db.add(time_log)
        db.commit()
        db.refresh(time_log)
        return time_log

    @staticmethod
    def get(db: Session, time_log_id: int) -> TimeLog:
        return db.query(TimeLog).filter(TimeLog.id == time_log_id).first()

    @staticmethod
    def update(db: Session, time_log_id: int, update_data: dict) -> TimeLog:
        time_log = db.query(TimeLog).filter(TimeLog.id == time_log_id).first()
        if time_log:
            for key, value in update_data.items():
                setattr(time_log, key, value)
            db.commit()
            db.refresh(time_log)
        return time_log

    @staticmethod
    def delete(db: Session, time_log_id: int) -> bool:
        time_log = db.query(TimeLog).filter(TimeLog.id == time_log_id).first()
        if time_log:
            db.delete(time_log)
            db.commit()
            return True
        return False

    @staticmethod
    def list(db: Session):
        return db.query(TimeLog).all()

    @staticmethod
    def list_by_chart_id(db: Session, chart_id: int):
        return db.query(TimeLog).filter(TimeLog.chart_id == chart_id).all()

    @staticmethod
    def start_timer(db: Session, chart_id: int, step_name: str):
        """타이머 시작"""
        time_log_data = {
            "chart_id": chart_id,
            "step_name": step_name,
            "start_time": datetime.now(),
            "created_at": datetime.now()
        }
        return TimeLogDAO.create(db, time_log_data)

    @staticmethod
    def end_timer(db: Session, time_log_id: int):
        """타이머 종료"""
        time_log = TimeLogDAO.get(db, time_log_id)
        if time_log and not time_log.end_time:
            end_time = datetime.now()
            elapsed_time = (end_time - time_log.start_time).total_seconds()
            update_data = {
                "end_time": end_time,
                "elapsed_time": elapsed_time
            }
            return TimeLogDAO.update(db, time_log_id, update_data)
        return time_log

    @staticmethod
    def get_total_time_by_chart(db: Session, chart_id: int) -> float:
        """차트의 전체 경과 시간 합계"""
        time_logs = TimeLogDAO.list_by_chart_id(db, chart_id)
        total_time = sum(log.elapsed_time or 0 for log in time_logs)
        return total_time
