import logging
from dotenv import load_dotenv

load_dotenv()
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_session, init_db
from .llm_client import LLMClient
from .models import ChatSession, Document, Message, MessageRole, RagConfig, RagStore, User
from .rag_engine import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    user_id: int
    text: str
    model: Optional[str] = None
    temperature: float = 0.2
    use_rag: bool = False
    store_name: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    used_rag: bool
    model: Optional[str]
    session_id: int


class RagInitRequest(BaseModel):
    user_id: int
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    vector_store: str = Field(default="faiss")


class HistoryResponse(BaseModel):
    messages: List[dict]


class ModelSelectRequest(BaseModel):
    user_id: int
    model: str
    use_rag: bool = False


class StoreCreateRequest(BaseModel):
    user_id: int
    name: str
    embedding_model: str = "all-MiniLM-L6-v2"


class StoreDeleteRequest(BaseModel):
    user_id: int
    name: str


class StoreSelectRequest(BaseModel):
    user_id: int
    name: Optional[str] = None  # None означает clear LLM


def create_app() -> FastAPI:
    app = FastAPI(title="Telegram LLM Bot Backend")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.rag_engine = RAGEngine()
    app.state.llm_client = LLMClient()

    @app.on_event("startup")
    async def _startup() -> None:
        await init_db()
        await app.state.rag_engine.ensure_base()
        logger.info("Application started")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await app.state.llm_client.aclose()

    async def _get_user(session: AsyncSession, telegram_id: int) -> User:
        result = await session.execute(select(User).where(User.telegram_id == telegram_id))
        user = result.scalars().first()
        if user:
            return user
        user = User(telegram_id=telegram_id)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user

    async def _get_config(session: AsyncSession, user: User) -> RagConfig:
        result = await session.execute(select(RagConfig).where(RagConfig.user_id == user.id))
        config = result.scalars().first()
        if not config:
            config = RagConfig(
                user_id=user.id,
                embedding_model="all-MiniLM-L6-v2",
                vector_store="faiss",
                store_path=str(app.state.rag_engine.base_path / str(user.telegram_id)),
                active_store_name=None,
            )
            session.add(config)
            await session.commit()
            await session.refresh(config)
        return config

    async def _get_store(session: AsyncSession, user: User, name: str) -> Optional[RagStore]:
        result = await session.execute(
            select(RagStore).where(RagStore.user_id == user.id, RagStore.name == name)
        )
        return result.scalars().first()

    async def _get_or_create_session(
        session: AsyncSession, user: User, model_name: Optional[str], use_rag: bool
    ) -> ChatSession:
        result = await session.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user.id)
            .order_by(ChatSession.created_at.desc())
        )
        chat_session = result.scalars().first()
        if chat_session and chat_session.model_name == model_name and chat_session.use_rag == use_rag:
            return chat_session
        chat_session = ChatSession(user_id=user.id, model_name=model_name, use_rag=use_rag)
        session.add(chat_session)
        await session.commit()
        await session.refresh(chat_session)
        return chat_session

    async def _recent_messages(session: AsyncSession, user: User, limit: int = 20) -> List[Message]:
        result = await session.execute(
            select(Message)
            .join(ChatSession, ChatSession.id == Message.session_id)
            .where(ChatSession.user_id == user.id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        return list(reversed(result.scalars().all()))

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/models")
    async def models() -> dict:
        models = await app.state.llm_client.list_models()
        return {"models": models}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        payload: ChatRequest, db: AsyncSession = Depends(get_session)
    ) -> ChatResponse:
        user = await _get_user(db, payload.user_id)
        config = await _get_config(db, user)

        store_name = payload.store_name or config.active_store_name
        use_rag = bool(store_name)

        chat_session = await _get_or_create_session(
            db, user, payload.model, use_rag
        )

        history = await _recent_messages(db, user, limit=20)
        messages = [{"role": m.role.value if hasattr(m.role, "value") else m.role, "content": m.content} for m in history]

        if use_rag and store_name:
            store = await _get_store(db, user, store_name)
            if store:
                contexts = await app.state.rag_engine.search(
                    user_id=user.telegram_id,
                    store_name=store_name,
                    query=payload.text,
                    embedding_model=store.embedding_model,
                )
                if contexts:
                    context_text = "\n\n".join([c.text for c in contexts])
                    messages.append(
                        {
                            "role": "system",
                            "content": f"Use the following context when answering:\n{context_text}",
                        }
                    )

        messages.append({"role": "user", "content": payload.text})

        model_name = payload.model or "default"
        try:
            reply = await app.state.llm_client.chat_completion(
                messages=messages, model=model_name, temperature=payload.temperature
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Chat completion failed: %s", exc)
            raise HTTPException(status_code=502, detail="LLM service unavailable") from exc

        db.add(
            Message(
                session_id=chat_session.id,
                role=MessageRole.USER,
                content=payload.text,
                model_name=model_name,
                used_rag=use_rag,
            )
        )
        db.add(
            Message(
                session_id=chat_session.id,
                role=MessageRole.ASSISTANT,
                content=reply,
                model_name=model_name,
                used_rag=use_rag,
            )
        )
        await db.commit()

        return ChatResponse(
            reply=reply,
            used_rag=use_rag,
            model=model_name,
            session_id=chat_session.id,
        )

    @app.post("/rag/init")
    async def rag_init(payload: RagInitRequest, db: AsyncSession = Depends(get_session)) -> dict:
        user = await _get_user(db, payload.user_id)
        config = await _get_config(db, user)
        config.embedding_model = payload.embedding_model
        config.vector_store = payload.vector_store
        await db.commit()
        return {"status": "ok", "embedding_model": config.embedding_model, "vector_store": config.vector_store}

    @app.get("/rag/stores/{user_id}")
    async def list_stores(user_id: int, db: AsyncSession = Depends(get_session)) -> dict:
        user = await _get_user(db, user_id)
        config = await _get_config(db, user)
        result = await db.execute(select(RagStore).where(RagStore.user_id == user.id))
        stores = [
            {"name": s.name, "embedding_model": s.embedding_model, "store_path": s.store_path}
            for s in result.scalars().all()
        ]
        return {"stores": stores, "active": config.active_store_name}

    @app.post("/rag/store/create")
    async def create_store(payload: StoreCreateRequest, db: AsyncSession = Depends(get_session)) -> dict:
        user = await _get_user(db, payload.user_id)
        config = await _get_config(db, user)
        exists = await _get_store(db, user, payload.name)
        if exists:
            return {"status": "exists", "name": payload.name}

        store_dir = app.state.rag_engine._store_dir(user.telegram_id, payload.name)
        store = RagStore(
            user_id=user.id,
            config_id=config.id,
            name=payload.name,
            store_path=str(store_dir),
            embedding_model=payload.embedding_model,
        )
        db.add(store)
        await db.commit()
        return {"status": "ok", "name": payload.name}

    @app.post("/rag/store/delete")
    async def delete_store(payload: StoreDeleteRequest, db: AsyncSession = Depends(get_session)) -> dict:
        import shutil

        user = await _get_user(db, payload.user_id)
        store = await _get_store(db, user, payload.name)
        if not store:
            return {"status": "not_found"}
        # remove dir on disk
        try:
            shutil.rmtree(store.store_path, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
        await db.execute(
            delete(Document).where(
                Document.user_id == user.id, Document.store_name == payload.name
            )
        )
        await db.delete(store)
        config = await _get_config(db, user)
        if config.active_store_name == payload.name:
            config.active_store_name = None
        await db.commit()
        return {"status": "deleted"}

    @app.post("/rag/store/select")
    async def select_store(payload: StoreSelectRequest, db: AsyncSession = Depends(get_session)) -> dict:
        user = await _get_user(db, payload.user_id)
        config = await _get_config(db, user)
        if payload.name:
            store = await _get_store(db, user, payload.name)
            if not store:
                raise HTTPException(status_code=404, detail="Store not found")
            config.active_store_name = payload.name
        else:
            config.active_store_name = None  # clear LLM mode
        await db.commit()
        return {"status": "ok", "active": config.active_store_name}

    @app.post("/rag/upload")
    async def rag_upload(
        user_id: int,
        file: UploadFile = File(...),
        embedding_model: str = "all-MiniLM-L6-v2",
        db: AsyncSession = Depends(get_session),
        store_name: Optional[str] = None,
    ) -> dict:
        import tempfile
        from pathlib import Path

        if not store_name:
            raise HTTPException(status_code=400, detail="store_name is required")

        content = await file.read()
        user = await _get_user(db, user_id)
        config = await _get_config(db, user)
        store = await _get_store(db, user, store_name)
        if not store:
            store_dir = app.state.rag_engine._store_dir(user.telegram_id, store_name)
            store = RagStore(
                user_id=user.id,
                config_id=config.id,
                name=store_name,
                store_path=str(store_dir),
                embedding_model=embedding_model,
            )
            db.add(store)
            await db.commit()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            chunks, index_path = await app.state.rag_engine.ingest_file(
                user_id=user_id, store_name=store_name, file_path=str(tmp_path), embedding_model=store.embedding_model
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        from .models import Document

        doc = Document(user_id=user.id, store_name=store_name, filename=file.filename, path=str(index_path))
        db.add(doc)
        await db.commit()
        return {"status": "ok", "chunks": chunks, "index_path": index_path}

    @app.post("/session/model")
    async def select_model(
        payload: ModelSelectRequest, db: AsyncSession = Depends(get_session)
    ) -> dict:
        user = await _get_user(db, payload.user_id)
        chat_session = ChatSession(
            user_id=user.id, model_name=payload.model, use_rag=payload.use_rag
        )
        db.add(chat_session)
        await db.commit()
        await db.refresh(chat_session)
        return {"status": "ok", "session_id": chat_session.id, "model": payload.model}

    @app.get("/history/{user_id}", response_model=HistoryResponse)
    async def history(user_id: int, limit: int = 10, db: AsyncSession = Depends(get_session)) -> HistoryResponse:
        result = await db.execute(
            select(Message)
            .join(ChatSession, ChatSession.id == Message.session_id)
            .join(User, User.id == ChatSession.user_id)
            .where(User.telegram_id == user_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        messages = [
            {
                "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                "content": msg.content,
                "model": msg.model_name,
                "used_rag": msg.used_rag,
                "created_at": msg.created_at,
            }
            for msg in result.scalars().all()
        ]
        return HistoryResponse(messages=list(reversed(messages)))

    @app.post("/clear_cache")
    async def clear_cache(user_id: int, db: AsyncSession = Depends(get_session)) -> dict:
        result = await db.execute(select(User).where(User.telegram_id == user_id))
        user = result.scalars().first()
        if user:
            subq = (
                select(ChatSession.id)
                .where(ChatSession.user_id == user.id)
                .subquery()
            )
            await db.execute(delete(Message).where(Message.session_id.in_(select(subq.c.id))))
            await db.commit()
        return {"status": "cleared"}

    return app


app = create_app()

