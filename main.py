from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler,
    MessageHandler, ContextTypes, filters
)

import fitz  # PyMuPDF

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter



# ─────────────────────────────────────────
# Load ENV
# ─────────────────────────────────────────
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "tel-notes"

# ─────────────────────────────────────────
# Pinecone Init (Modern)
# ─────────────────────────────────────────
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(INDEX_NAME)

# ─────────────────────────────────────────
# HuggingFace Embedding Model
# ─────────────────────────────────────────
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vectorstore = PineconeVectorStore(index=pinecone_index, embedding=embedding)

# ─────────────────────────────────────────
# Text Splitter
# ─────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

# ─────────────────────────────────────────
# Logic
# ─────────────────────────────────────────
async def search(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query_text = " ".join(context.args)
    if not query_text:
        await update.message.reply_text("⚠️ Usage: /search <your query>")
        return

    results = vectorstore.similarity_search(query_text, k=3)

    if not results:
        await update.message.reply_text("❌ No results found.")
        return

    response = ""
    for doc in results:
        content = doc.page_content
        match_index = content.lower().find(query_text.lower())
        if match_index != -1:
            snippet_start = max(match_index - 50, 0)
            snippet_end = min(match_index + 250, len(content))
            snippet = content[snippet_start:snippet_end].strip()
            response += f"🔎 ...{snippet}...\n\n"
        else:
            # fallback if no direct match, just add first 200 chars
            response += f"📝 {content[:200].strip()}...\n\n"

    # Telegram message limit is 4096 characters
    for i in range(0, len(response), 4000):
        await update.message.reply_text(response[i:i+4000])


async def add_note(update: Update, context: ContextTypes.DEFAULT_TYPE):
    note_text = " ".join(context.args)
    if not note_text:
        await update.message.reply_text("⚠️ Usage: /add <your note>")
        return

    splits = text_splitter.split_documents([Document(page_content=note_text)])
    vectorstore.add_documents(splits)
    await update.message.reply_text("✅ Note indexed.")

async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file = update.message.document
    if file.mime_type != "application/pdf":
        await update.message.reply_text("⚠️ Only PDF files are supported.")
        return

    telegram_file = await file.get_file()
    file_path = f"./{file.file_name}"
    await telegram_file.download_to_drive(file_path)

    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()

    if not text.strip():
        await update.message.reply_text("❌ Failed to extract any text from the PDF.")
        return

    splits = text_splitter.split_documents([Document(page_content=text)])
    print(f"📄 Number of chunks: {len(splits)}")
    for i, s in enumerate(splits):
        print(f"🔹 Chunk {i+1}: {s.page_content[:80]}...")

    vectorstore.add_documents(splits)
    await update.message.reply_text(f"✅ PDF '{file.file_name}' indexed successfully.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("➕ Add Note", callback_data="add_note")],
        [InlineKeyboardButton("🔍 Search Notes", callback_data="search_notes")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("🤖 Welcome!", reply_markup=reply_markup)

async def handle_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "add_note":
        await query.message.reply_text("Send: `/add your note here`", parse_mode="Markdown")
    elif query.data == "search_notes":
        await query.message.reply_text("Send: `/search your query`", parse_mode="Markdown")

# ─────────────────────────────────────────
# Telegram Bot Run
# ─────────────────────────────────────────
if __name__ == "__main__":
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("add", add_note))
    app.add_handler(CommandHandler("search", search))
    app.add_handler(CallbackQueryHandler(handle_buttons))
    app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))

    print("🤖 Bot running with HuggingFace + LangChain + Pinecone")
    app.run_polling()
