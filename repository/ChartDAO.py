from sqlalchemy.orm import Session
from database.schema import Chart

class ChartDAO:
    
    @staticmethod
    def create(db: Session, chart_data: dict) -> Chart:
        chart = Chart(**chart_data)
        db.add(chart)
        db.commit()
        db.refresh(chart)
        return chart

    @staticmethod
    def get(db: Session, chart_id: int) -> Chart:
        return db.query(Chart).filter(Chart.uid == chart_id).first()

    @staticmethod
    def update(db: Session, chart_id: int, update_data: dict) -> Chart:
        chart = db.query(Chart).filter(Chart.uid == chart_id).first()
        if chart:
            for key, value in update_data.items():
                setattr(chart, key, value)
            db.commit()
            db.refresh(chart)
        return chart

    @staticmethod
    def delete(db: Session, chart_id: int) -> bool:
        chart = db.query(Chart).filter(Chart.uid == chart_id).first()
        if chart:
            db.delete(chart)
            db.commit()
            return True
        return False

    @staticmethod
    def list(db: Session):
        return db.query(Chart).all()
