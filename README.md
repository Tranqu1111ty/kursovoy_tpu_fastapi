# Telegram LLM Bot with RAG (FastAPI + aiogram + LM Studio)

Локальный Telegram-бот с Retrieval-Augmented Generation. Бэкенд на FastAPI, бот на aiogram, хранение метаданных в PostgreSQL, векторные индексы на диске (FAISS + sentence-transformers), интеграция с LM Studio (OpenAI-совместимый API).

## Возможности
- Команды бота: `/start`, `/models` (выбор LLM), `/rag` (создание/выбор/удаление базы), `/clear` (очистка истории), `/history`.
- RAG с именованными векторными базами (на пользователя): загрузка PDF/TXT/DOCX, поиск контекста в FAISS, переключение режимов RAG ↔ чистая LLM.
- Все LLM-вызовы и RAG-операции проходят через FastAPI (логируемый слой).

## Требования
- Python 3.10+
- PostgreSQL (доступный через `DATABASE_URL`)
- LM Studio, доступный по `LM_STUDIO_BASE` (по умолчанию `http://192.168.0.102:1234`)

## Установка
```bash
python -m venv .venv
. .venv/Scripts/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Переменные окружения (.env)
```
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/llm_bot
LM_STUDIO_BASE=http://192.168.0.102:1234
API_URL=http://localhost:8000
TELEGRAM_BOT_TOKEN=ваш_токен
```

## Запуск
В одном окне:
```bash
uvicorn telegram_llm_bot.app.main:app --reload
```
В другом:
```bash
python -m telegram_llm_bot.app.bot
```

## Структура
```
telegram_llm_bot/
  app/
    main.py        # FastAPI эндпоинты
    bot.py         # aiogram-бот
    models.py      # SQLAlchemy модели (users, sessions, messages, rag_config, rag_stores, documents)
    rag_engine.py  # FAISS + sentence-transformers, ingest/search
    llm_client.py  # LM Studio OpenAI-совместимый клиент
    database.py    # asyncpg + SQLAlchemy engine/session
```

## RAG и хранение данных
- Метаданные (пользователи, сессии, сообщения, rag_config, rag_stores, documents) — в PostgreSQL.
- Векторные индексы и чанки — на диске: `./vector_stores/{user_id}/{store_name}/index.faiss` и `chunks.json`.

## Быстрые подсказки
- Создать новую RAG-базу: `/rag` → «Создать новую базу», отправить документ с подписью = имя базы.
- Выбрать базу: `/rag` → «Выбрать базу».
- Удалить базу: `/rag` → «Удалить базу».
- Отключить RAG (чистая LLM): `/rag` → «Clear LLM».

## Лицензия
Добавьте подходящую лицензию при публикации (MIT/Apache-2.0 и т.п.).

