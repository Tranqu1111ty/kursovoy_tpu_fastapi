import asyncio
import logging
import os
from typing import Dict, Optional

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is required")

API_URL = os.getenv("API_URL", "http://localhost:8000")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
http_client = httpx.AsyncClient(base_url=API_URL, timeout=60)

user_state: Dict[int, Dict[str, Optional[str]]] = {}


async def fetch_models() -> list[str]:
    resp = await http_client.get("/models")
    resp.raise_for_status()
    return resp.json().get("models", [])


async def select_model(user_id: int, model: str) -> None:
    payload = {"user_id": user_id, "model": model}
    resp = await http_client.post("/session/model", json=payload)
    resp.raise_for_status()
    user_state[user_id] = {"model": model}
    # keep current store if exists
    if "store" not in user_state[user_id]:
        user_state[user_id]["store"] = None


@dp.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(
        "Привет! Я локальный LLM-бот.\n"
        "/models — список и выбор модели\n"
        "/rag — загрузка документов и настройка RAG\n"
        "/clear — очистка кэша переписки\n"
        "/history — последние сообщения\n"
        "Просто отправь сообщение, чтобы получить ответ."
    )


@dp.message(Command("models"))
async def models(message: Message) -> None:
    try:
        models = await fetch_models()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Model fetch failed: %s", exc)
        await message.answer("Не удалось получить список моделей.")
        return

    if not models:
        await message.answer("Модели не найдены. Проверь LM Studio.")
        return

    buttons = [
        [InlineKeyboardButton(text=m, callback_data=f"set_model:{m}")] for m in models
    ]
    await message.answer("Выбери модель:", reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons))


@dp.callback_query(F.data.startswith("set_model:"))
async def set_model(query: CallbackQuery) -> None:
    model = query.data.split(":", 1)[1]
    try:
        await select_model(query.from_user.id, model)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Model select failed: %s", exc)
        await query.message.answer("Не удалось выбрать модель.")
        await query.answer()
        return

    await query.message.answer(f"Модель установлена: {model}. История кэша очищена.")
    await query.answer()


@dp.message(Command("rag"))
async def rag(message: Message) -> None:
    kb = InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Создать новую базу", callback_data="rag_menu:create")],
            [InlineKeyboardButton(text="Выбрать базу", callback_data="rag_menu:select")],
            [InlineKeyboardButton(text="Удалить базу", callback_data="rag_menu:delete")],
            [InlineKeyboardButton(text="Clear LLM (без RAG)", callback_data="rag_menu:clear_llm")],
        ]
    )
    await message.answer("Выбери действие для RAG:", reply_markup=kb)


@dp.callback_query(F.data.startswith("rag_menu:"))
async def rag_menu(query: CallbackQuery) -> None:
    action = query.data.split(":", 1)[1]
    uid = query.from_user.id
    if action == "create":
        user_state.setdefault(uid, {})["pending"] = "create_store"
        await query.message.answer(
            "Отправь документ (PDF/TXT/DOCX). Подпись к файлу = название новой векторной базы."
        )
    elif action == "select":
        resp = await http_client.get(f"/rag/stores/{uid}")
        stores = resp.json().get("stores", [])
        if not stores:
            await query.message.answer("Нет доступных баз. Создай новую.")
        else:
            buttons = [
                [InlineKeyboardButton(text=s["name"], callback_data=f"rag_select:{s['name']}")]
                for s in stores
            ]
            await query.message.answer("Выбери базу для RAG:", reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons))
    elif action == "delete":
        resp = await http_client.get(f"/rag/stores/{uid}")
        stores = resp.json().get("stores", [])
        if not stores:
            await query.message.answer("Нет баз для удаления.")
        else:
            buttons = [
                [InlineKeyboardButton(text=f"Удалить {s['name']}", callback_data=f"rag_delete:{s['name']}")]
                for s in stores
            ]
            await query.message.answer("Выбери базу для удаления:", reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons))
    elif action == "clear_llm":
        await http_client.post("/rag/store/select", json={"user_id": uid, "name": None})
        state = user_state.setdefault(uid, {})
        state["store"] = None
        await query.message.answer("Режим LLM без RAG активирован.")
    await query.answer()


@dp.callback_query(F.data.startswith("rag_select:"))
async def rag_select(query: CallbackQuery) -> None:
    name = query.data.split(":", 1)[1]
    uid = query.from_user.id
    await http_client.post("/rag/store/select", json={"user_id": uid, "name": name})
    state = user_state.setdefault(uid, {})
    state["store"] = name
    await query.message.answer(f"Активирована база: {name}. Чат работает в режиме RAG.")
    await query.answer()


@dp.callback_query(F.data.startswith("rag_delete:"))
async def rag_delete(query: CallbackQuery) -> None:
    name = query.data.split(":", 1)[1]
    uid = query.from_user.id
    await http_client.post("/rag/store/delete", json={"user_id": uid, "name": name})
    state = user_state.setdefault(uid, {})
    if state.get("store") == name:
        state["store"] = None
    await query.message.answer(f"База {name} удалена.")
    await query.answer()


@dp.message(F.document)
async def handle_document(message: Message) -> None:
    state = user_state.setdefault(message.from_user.id, {})
    pending = state.get("pending")
    if pending != "create_store":
        await message.answer("Сначала выбери действие /rag -> Создать новую базу и пришли файл с названием в подписи.")
        return
    store_name = (message.caption or "").strip()
    if not store_name:
        await message.answer("Укажи название базы в подписи к файлу.")
        return
    embedding_model = "all-MiniLM-L6-v2"
    file_info = await bot.get_file(message.document.file_id)
    file_bytes = await bot.download_file(file_info.file_path)
    files = {"file": (message.document.file_name, file_bytes.read())}
    data = {"user_id": str(message.from_user.id), "embedding_model": embedding_model, "store_name": store_name}
    try:
        resp = await http_client.post("/rag/upload", files=files, params=data)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG upload failed: %s", exc)
        await message.answer("Не удалось обработать документ.")
        return
    await http_client.post("/rag/store/select", json={"user_id": message.from_user.id, "name": store_name})
    state["store"] = store_name
    state.pop("pending", None)
    result = resp.json()
    chunks = result.get("chunks", 0)
    await message.answer(f"База '{store_name}' создана. Документ добавлен. Чанков: {chunks}. Режим RAG активен.")


@dp.message(Command("clear"))
async def clear_cache(message: Message) -> None:
    try:
        await http_client.post("/clear_cache", params={"user_id": message.from_user.id})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Cache clear failed: %s", exc)
        await message.answer("Не удалось очистить кэш.")
        return
    await message.answer("Кэш переписки очищен. RAG-артефакты сохранены.")


@dp.message(Command("history"))
async def history(message: Message) -> None:
    try:
        resp = await http_client.get(f"/history/{message.from_user.id}", params={"limit": 10})
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.exception("History fetch failed: %s", exc)
        await message.answer("Не удалось получить историю.")
        return
    messages = resp.json().get("messages", [])
    if not messages:
        await message.answer("История пуста.")
        return
    text = "\n\n".join(
        f"{m['role']}: {m['content']}" for m in messages[-10:]
    )
    await message.answer(text[:4000])


@dp.message()
async def chat(message: Message) -> None:
    state = user_state.get(message.from_user.id, {})
    model = state.get("model")
    store = state.get("store")
    payload = {
        "user_id": message.from_user.id,
        "text": message.text,
        "model": model,
        "use_rag": bool(store),
        "store_name": store,
    }
    try:
        resp = await http_client.post("/chat", json=payload)
        resp.raise_for_status()
        reply = resp.json().get("reply", "")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Chat failed: %s", exc)
        await message.answer("Ошибка при обращении к модели.")
        return
    await message.answer(reply[:4000])


async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

