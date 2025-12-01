import matplotlib.pyplot as plt
import numpy as np

sv_data = """Programming Language,Min,Q1,Median,Q3,Max,Outlier
Python,100,200,500,800,300,[1000]
Java,200,400,600,900,400,[1200]
JavaScript,150,300,450,700,325,[1100]
C++,175,350,625,850,425,[1300]
Ruby,125,250,550,750,350,[1150]"""

# Parse the CSV data
lines = sv_data.split('\n')
header = lines[0].split(',')
data = [line.split(',') for line in lines[1:]]

languages = [row[0] for row in data]
min_values = [int(row[1]) for row in data]
q1_values = [int(row[2]) for row in data]
median_values = [int(row[3]) for row in data]
q3_values = [int(row[4]) for row in data]
max_values = [int(row[5]) for row in data]

# Set up the figure and axes
fig, ax1 = plt.subplots(figsize=(10, 8))

# Plot the bar chart
bar_width = 0.4
index = np.arange(len(languages))

bars1 = ax1.bar(index - bar_width/2, min_values, bar_width, label='Min', color='#48D1CC')
bars2 = ax1.bar(index + bar_width/2, max_values, bar_width, label='Max', color='#6B8E23')

# Set up a secondary axis for the median line plot
ax2 = ax1.twinx()
line1 = ax2.plot(index, median_values, color='r', marker='o', label='Median', linestyle='-', linewidth=2)

# Add data labels to the bar chart
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}', ha='center', va='bottom', fontsize=10, color='black')

for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}', ha='center', va='bottom', fontsize=10, color='black')

# Add data labels to the line plot
for i, txt in enumerate(median_values):
    ax2.annotate(txt, (index[i], median_values[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8, color='r')

# Customize the appearance
ax1.set_xlabel('Programming Languages', fontsize=12)
ax1.set_ylabel('Count of Values', fontsize=16, color='black')
ax2.set_ylabel('Median Values', fontsize=16, color='black')
ax1.set_title('Programming Language Data', fontsize=18)

# Set the ticks and gridlines for both axes
ax1.set_xticks(index)
ax1.set_xticklabels(languages, fontsize=10)
ax1.grid(True, linestyle='-', linewidth=0.7)
ax2.grid(False)

# Add legends for both bar and line data
bars_legend = ax1.legend(loc='upper left', bbox_to_anchor=(-0.1, 1.1))
lines_legend = ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('19182611262.png', bbox_inches='tight', dpi=80)

import datetime
from sqlalchemy.orm import Session
from database.conn import SessionLocal
from repository.ChartDAO import ChartDAO
from repository.QuestionDAO import QuestionDAO

def test_chart_crud():
    db: Session = SessionLocal()
    # Create
    chart_data = {
        "uid": 1,
        "type": "bar",
        "query_count": 0,
        "total_time": 0
    }
    chart = ChartDAO.create(db, chart_data)
    assert chart.id is not None
    print("Create OK", chart)

    # Read
    chart_read = ChartDAO.get(db, chart.id)
    assert chart_read is not None
    print("Read OK", chart_read)

    # Update
    update_data = {"query_count": 5, "total_time": 123}
    chart_updated = ChartDAO.update(db, chart.id, update_data)
    assert chart_updated.query_count == 5
    print("Update OK", chart_updated)

    # List
    charts = ChartDAO.list(db)
    assert len(charts) > 0
    print("List OK", charts)

    # Delete
    result = ChartDAO.delete(db, chart.id)
    assert result is True
    print("Delete OK", result)

    db.close()

def test_question_crud():
    db: Session = SessionLocal()
    # Chart 먼저 생성
    chart_data = {
        "uid": 2,
        "type": "pie",
        "query_count": 0,
        "total_time": 0
    }
    chart = ChartDAO.create(db, chart_data)

    # Create Question
    now = datetime.datetime.now()
    question_data = {
        "chart_id": chart.id,
        "content": "차트에 대한 질문입니다.",
        "create_date": now,
        "answer_content": None,
        "answer_create_date": None
    }
    question = QuestionDAO.create(db, question_data)
    assert question.id is not None
    print("Question Create OK", question)

    # Read
    question_read = QuestionDAO.get(db, question.id)
    assert question_read is not None
    print("Question Read OK", question_read)

    # Update
    update_data = {"answer_content": "답변입니다.", "answer_create_date": now}
    question_updated = QuestionDAO.update(db, question.id, update_data)
    assert question_updated.answer_content == "답변입니다."
    print("Question Update OK", question_updated)

    # List
    questions = QuestionDAO.list(db)
    assert len(questions) > 0
    print("Question List OK", questions)

    # Delete
    result = QuestionDAO.delete(db, question.id)
    assert result is True
    print("Question Delete OK", result)

    # 차트도 삭제
    ChartDAO.delete(db, chart.id)

    # 차트 id로 해당 질문/답변 리스트 조회
    questions_by_chart = QuestionDAO.list_by_chart_id(db, chart.id)
    assert any(q.id == question.id for q in questions_by_chart)
    print("Questions by Chart ID OK", questions_by_chart)
    
    db.close()

def test_insert_by_chart_id():
    db: Session = SessionLocal()
    # Chart 생성
    chart_data = {
        "uid": 3,
        "type": "line",
        "query_count": 0,
        "total_time": 0
    }
    chart = ChartDAO.create(db, chart_data)

    # 질문/답변 생성
    now = datetime.datetime.now()
    question = QuestionDAO.insert_by_chart_id(
        db,
        chart_id=chart.id,
        content="차트에 대한 새로운 질문입니다.",
        create_date=now,
        answer_content="새로운 답변입니다.",
        answer_create_date=now
    )
    assert question.id is not None
    assert question.chart_id == chart.id
    assert question.content == "차트에 대한 새로운 질문입니다."
    assert question.answer_content == "새로운 답변입니다."
    print("Insert by Chart ID OK", question)

    # 정리
    QuestionDAO.delete(db, question.id)
    ChartDAO.delete(db, chart.id)
    db.close()

if __name__ == "__main__":
    test_chart_crud()
    # test_question_crud()
    # test_insert_by_chart_id()
