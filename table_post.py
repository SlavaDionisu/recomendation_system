from database import Base, SessionLocal
from sqlalchemy import Column, Integer, String

class Post(Base):
    __tablename__ = "post"
    __table_args__ = {"schema": "public"}
    id = Column(Integer, primary_key=True)
    text = Column(String)
    topic = Column(String)

if __name__ == "__main__":
    session = SessionLocal()
    business_posts = (
        session.query(Post)
        .filter(Post.topic == "business")
        .order_by(Post.id.desc())
        .limit(10)
        .all()
    )
    ids = [post.id for post in business_posts]
    print(ids)