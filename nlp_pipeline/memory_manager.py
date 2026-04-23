"""
Memory Manager for Legal AI
Handles short-term session memory in SQLite and long-term semantic memory in
ChromaDB. It stores recent turns, retrieved documents, and compact summaries
that can be recalled across sessions for the same user.
"""

import sqlite3
import os
import json
import uuid
import time

class MemoryManager:
    def __init__(self, chroma_client, embedder, groq_client, db_path="chat_history.db"):
        self.chroma_client = chroma_client
        self.embedder = embedder
        self.groq_client = groq_client
        
        # Initialize SQLite for short-term memory and interaction audit trails.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(base_dir, db_path)
        self._init_sqlite()
        
        # Initialize ChromaDB for long-term memory
        self.collection_name = "user-memory"
        try:
            self.memory_collection = self.chroma_client.get_collection(self.collection_name)
        except Exception:
            self.memory_collection = self.chroma_client.create_collection(
                name=self.collection_name, 
                metadata={"hnsw:space": "cosine"}
            )
            
    def _init_sqlite(self):
        """Sets up SQLite tables for chat history and structured memory events."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    session_id TEXT,
                    query TEXT,
                    answer TEXT,
                    retrieved_docs_json TEXT,
                    intermediate_reasoning TEXT,
                    success_score REAL DEFAULT 1.0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            c.execute('CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id, id)')

    def add_message(self, session_id: str, role: str, content: str):
        """Adds a message to the short-term memory (SQLite)."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                'INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)',
                (session_id, role, content)
            )

    def get_short_term_memory(self, session_id: str, limit: int = 5) -> list:
        """Retrieves the last N messages for a given session."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # Get the last N messages, then reverse them so they are chronological.
            c.execute('''
                SELECT role, content FROM messages
                WHERE session_id = ?
                ORDER BY id DESC LIMIT ?
            ''', (session_id, limit))
            rows = c.fetchall()
        
        # Reverse to return chronological order (oldest -> newest in the recent window).
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def record_interaction(
        self,
        user_id: str,
        session_id: str,
        query: str,
        answer: str,
        retrieved_docs: list,
        intermediate_reasoning: str = "",
        success_score: float = 1.0,
    ):
        """
        Stores a structured short-term/audit memory entry.
        retrieved_docs should be JSON-serializable citation metadata.
        """
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                '''
                INSERT INTO interactions (
                    user_id, session_id, query, answer, retrieved_docs_json,
                    intermediate_reasoning, success_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    user_id,
                    session_id,
                    query,
                    answer,
                    json.dumps(retrieved_docs, ensure_ascii=True),
                    intermediate_reasoning,
                    success_score,
                )
            )

    def get_frequent_questions(self, user_id: str, limit: int = 5) -> list:
        """Returns the user's most frequently repeated legal questions."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                '''
                SELECT query, COUNT(*) AS frequency
                FROM interactions
                WHERE user_id = ?
                GROUP BY query
                ORDER BY frequency DESC, MAX(id) DESC
                LIMIT ?
                ''',
                (user_id, limit)
            )
            rows = c.fetchall()
        return [{"query": row[0], "frequency": row[1]} for row in rows]

    def summarize_and_store_long_term(
        self,
        user_id: str,
        session_id: str,
        user_query: str,
        ai_response: str,
        retrieved_docs: list | None = None,
        intermediate_reasoning: str = "",
    ):
        """
        Background task: summarizes a user query, retrieved context, and answer,
        then stores it in ChromaDB as a long-term memory vector.
        """
        try:
            retrieved_docs = retrieved_docs or []
            retrieved_context = "\n".join(
                f"- {doc.get('task', 'unknown')}: {doc.get('text', '')}"
                for doc in retrieved_docs[:3]
            )

            system_prompt = (
                "You are an AI Memory Summarizer. Extract the core facts from this interaction. "
                "Output ONLY a 1-2 sentence summary of what was asked, which legal context mattered, "
                "and what answer was useful. Focus on reusable legal concepts, user preferences, "
                "and successful reasoning steps. NO conversational filler."
            )
            
            completion = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"User: {user_query}\n\n"
                            f"Retrieved legal context:\n{retrieved_context or 'None'}\n\n"
                            f"Intermediate reasoning:\n{intermediate_reasoning or 'None'}\n\n"
                            f"AI: {ai_response}"
                        )
                    }
                ],
                temperature=0.1
            )
            
            summary = completion.choices[0].message.content.strip()
            
            # 2. Embed the summary
            vector = self.embedder.encode([summary], show_progress_bar=False)[0].tolist()
            
            # 3. Save to Chroma
            mem_id = f"mem_{user_id}_{session_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
            self.memory_collection.add(
                ids=[mem_id],
                embeddings=[vector],
                documents=[summary],
                metadatas=[{
                    "user_id": user_id,
                    "session_id": session_id,
                    "timestamp": time.time(),
                    "query": user_query[:500],
                }]
            )
            
        except Exception as e:
            print(f"Error saving long-term memory: {e}")

    def get_long_term_memory(self, user_id: str, current_query: str, top_k: int = 3) -> str:
        """
        Searches the Chroma collection for past summarized interactions 
        that are semantically relevant to the current query.
        Filters specifically for the current user so memory survives across sessions.
        """
        if self.memory_collection.count() == 0:
            return ""
            
        try:
            q_vec = self.embedder.encode([current_query], show_progress_bar=False)[0].tolist()
            
            # We filter by session_id to only retrieve this user's memories
            results = self.memory_collection.query(
                query_embeddings=[q_vec],
                n_results=top_k,
                where={"user_id": user_id}
            )
            
            if not results["documents"][0]:
                return ""
                
            memories = "\n".join([f"- {doc}" for doc in results["documents"][0]])
            return memories
            
        except Exception as e:
            print(f"Error retrieving long-term memory: {e}")
            return ""
