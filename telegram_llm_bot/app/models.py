import enum
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    sessions: Mapped[List["ChatSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    documents: Mapped[List["Document"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    rag_config: Mapped[Optional["RagConfig"]] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="cascade"))
    model_name: Mapped[str] = mapped_column(String(200), nullable=True)
    temperature: Mapped[float] = mapped_column(Float, default=0.2)
    use_rag: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="sessions")
    messages: Mapped[List["Message"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="cascade")
    )
    role: Mapped[MessageRole] = mapped_column(Enum(MessageRole))
    content: Mapped[str] = mapped_column(Text)
    model_name: Mapped[Optional[str]] = mapped_column(String(200))
    used_rag: Mapped[bool] = mapped_column(Boolean, default=False)
    meta: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    session: Mapped["ChatSession"] = relationship(back_populates="messages")


class RagConfig(Base):
    __tablename__ = "rag_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="cascade"))
    embedding_model: Mapped[str] = mapped_column(String(200), default="all-MiniLM-L6-v2")
    vector_store: Mapped[str] = mapped_column(String(50), default="faiss")
    store_path: Mapped[str] = mapped_column(String(500), default="")
    active_store_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="rag_config")
    stores: Mapped[List["RagStore"]] = relationship(
        back_populates="config", cascade="all, delete-orphan"
    )


class RagStore(Base):
    __tablename__ = "rag_stores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="cascade"))
    config_id: Mapped[int] = mapped_column(ForeignKey("rag_configs.id", ondelete="cascade"))
    name: Mapped[str] = mapped_column(String(200))
    store_path: Mapped[str] = mapped_column(String(500))
    embedding_model: Mapped[str] = mapped_column(String(200), default="all-MiniLM-L6-v2")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    config: Mapped["RagConfig"] = relationship(back_populates="stores")
    user: Mapped["User"] = relationship()


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="cascade"))
    store_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    filename: Mapped[str] = mapped_column(String(255))
    path: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="documents")

