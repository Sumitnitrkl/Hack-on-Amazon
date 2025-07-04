import sqlite3
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    """Initialize SQLite database to store extracted texts, metadata, and timestamps."""
    try:
        conn = sqlite3.connect("nalco_chatbot.db")
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                file_name TEXT PRIMARY KEY,
                extracted_text TEXT,
                timestamp TEXT
            )
            """
        )
        conn.commit()
        logger.info("Initialized SQLite database: nalco_chatbot.db")
    except sqlite3.Error as e:
        logger.error(f"Database initialization failed: {e}")
    finally:
        conn.close()

def store_document(file_name: str, extracted_text: str, timestamp: str = None):
    """Store the extracted text, metadata, and timestamp in SQLite database."""
    try:
        conn = sqlite3.connect("nalco_chatbot.db")
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO documents (file_name, extracted_text, timestamp) VALUES (?, ?, ?)",
            (file_name, extracted_text, timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        )
        conn.commit()
        logger.info(f"Stored document in SQLite: {file_name}")
    except sqlite3.Error as e:
        logger.error(f"Failed to store document {file_name}: {e}")
    finally:
        conn.close()

def load_documents_from_db():
    """Load all documents, their extracted texts, and timestamps from SQLite database."""
    documents = []
    try:
        conn = sqlite3.connect("nalco_chatbot.db")
        c = conn.cursor()
        c.execute("SELECT file_name, extracted_text, timestamp FROM documents")
        documents = c.fetchall()
        logger.info(f"Loaded {len(documents)} documents from SQLite database.")
    except sqlite3.Error as e:
        logger.error(f"Failed to load documents from database: {e}")
    finally:
        conn.close()
    return documents

if __name__ == "__main__":
    init_db()
