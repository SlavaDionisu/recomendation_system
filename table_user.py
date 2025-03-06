from database import Base, SessionLocal
from sqlalchemy import Column, Integer, String, func

class User(Base):
    __tablename__ = "user"
    __table_args__ = {"schema": "public"}
    id = Column(Integer, primary_key=True)
    gender = Column(Integer)
    age = Column(Integer)
    country = Column(String)
    city = Column(String)
    exp_group = Column(Integer)
    os = Column(String)
    source = Column(String)

if __name__ == "__main__":
    session = SessionLocal()
    result_table = (
        session.query(User.country, User.os, func.count().label('count')) #выбираем столбцы
        .filter(User.exp_group == 3) #эксп группа 3
        .group_by(User.country, User.os) #группитруем
        .having(func.count() > 100) #агрегируем по количеству и сразу фильтруем больше 100
        .order_by(func.count().desc()) #сортируем
        .all()
    )
    result_list = [(country, os, count) for country, os, count in result_table]
    print(result_list)