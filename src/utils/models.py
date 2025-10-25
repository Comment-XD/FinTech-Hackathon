from .database import Base
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, func


class Transactions(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    nameOrig = Column(String, index=True, nullable=False)
    type = Column(String, nullable=False)
    amount = Column(Float, nullable=False)
    oldbalanceOrg = Column(Float, nullable=True)
    newbalanceOrig = Column(Float, nullable=True)
    nameDest = Column(String, index=True, nullable=False)
    oldbalanceDest = Column(Float, nullable=True)
    newbalanceDest = Column(Float, nullable=True)
    isFraud = Column(Boolean, nullable=True, default=False)

    def __repr__(self):
        return (
            f"<Transaction id={self.id} nameOrig={self.nameOrig} "
            f"nameDest={self.nameDest} amount={self.amount} isFraud={self.isFraud}>"
        )