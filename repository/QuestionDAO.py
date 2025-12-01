from sqlalchemy.orm import Session
from database.schema import Question

class QuestionDAO:
    @staticmethod
    def create(db: Session, question_data: dict) -> Question:
        question = Question(**question_data)
        db.add(question)
        db.commit()
        db.refresh(question)
        return question

    @staticmethod
    def get(db: Session, question_id: int) -> Question:
        return db.query(Question).filter(Question.id == question_id).first()

    @staticmethod
    def update(db: Session, question_id: int, update_data: dict) -> Question:
        question = db.query(Question).filter(Question.id == question_id).first()
        if question:
            for key, value in update_data.items():
                setattr(question, key, value)
            db.commit()
            db.refresh(question)
        return question

    @staticmethod
    def delete(db: Session, question_id: int) -> bool:
        question = db.query(Question).filter(Question.id == question_id).first()
        if question:
            db.delete(question)
            db.commit()
            return True
        return False

    @staticmethod
    def list(db: Session):
        return db.query(Question).all()

    @staticmethod
    def list_by_chart_id(db: Session, chart_id: int):
        return db.query(Question).filter(Question.chart_id == chart_id).all()

    @staticmethod
    def insert_by_chart_id(db: Session, chart_id: int, content: str, create_date, answer_content=None, answer_create_date=None):
        question = Question(
            chart_id=chart_id,
            content=content,
            create_date=create_date,
            answer_content=answer_content,
        )
        db.add(question)
        db.commit()
        db.refresh(question)
        return question
