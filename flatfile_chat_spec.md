# FlatFile Chat Database Specification

## 🔍 Objective

Design a modular, extensible local-first chat storage system for AI-driven assistants that:

- Stores all user data, sessions, chat messages, documents, and media in the file system.
- Requires no external database or setup for initial use.
- Can be extended with additional data types (e.g. personas, embeddings, PDFs, images).
- Is **intentionally structured** to support future migration to a relational or vector database backend.

This system will grow from a minimal single-user chat session manager into a full-featured multi-user assistant memory system.

---

## 📁 Directory Structure

### Phase 1: Single User, Flat File Sessions

```
MyAIApp/
├── profile.json               # Single user profile
├── session_index.json         # Index of all sessions
├── sessions/
│   ├── session_20240722_1/
│   │   ├── session_meta.json
│   │   ├── messages.jsonl
│   │   └── (future: embeddings.npy, persona.json, etc.)
```

### Phase 3+: Multi-User Support

```
MyAIApp/
├── users/
│   ├── user_001/
│   │   ├── profile.json
│   │   ├── session_index.json
│   │   └── sessions/
│   │       ├── session_20240722_1/
│   │       │   ├── session_meta.json
│   │       │   ├── messages.jsonl
│   │       │   ├── document.pdf
│   │       │   ├── image1.jpg
│   │       │   ├── vector_index.jsonl
│   │       │   └── embeddings.npy
```

---

## 📦 File Types (Modular Expansion)

| File                         | Description                              | Phase |
| ---------------------------- | ---------------------------------------- | ----- |
| `profile.json`               | Stores user metadata & preferences       | 1     |
| `session_meta.json`          | Metadata about a specific chat session   | 1     |
| `messages.jsonl`             | Full chat history for a session          | 1     |
| `persona.json`               | Chatbot personality/prompt profile       | 2     |
| `context_history.json`       | Threaded memory / long-term context      | 3     |
| `vector_index.jsonl`         | Index of vector chunks (text/image)      | 3     |
| `embeddings.npy`             | Numpy float32 array of embedding vectors | 3     |
| `document.pdf` / `image.jpg` | Uploaded user files for analysis         | 2     |
| `audio.wav`, `video.mp4`     | Rich media files (future multimodal)     | 4+    |
| `summary.json`               | Summary/recap of session                 | 3+    |

---

## 🪜 Development Phases

### ✅ Phase 1: Basic Session Framework

- FlatFileSessionManager
- Create/load session directories
- Save/load `messages.jsonl` and `session_meta.json`
- Simple CLI or test harness to run one session

### ✅ Phase 2: Extend to Files and Personas

- Add file upload and copying into session folder
- Add support for saving persona configuration to `persona.json`
- Link messages to uploaded files in metadata

### ✅ Phase 3: Add Vectors and Context

- Add embedding vector support: `vector_index.jsonl`, `embeddings.npy`
- Chunking text and images for semantic search
- Add support for `context_history.json` (cross-session recall)

### 🔒 Phase 4: Multi-User System

- Introduce `FlatFileUserManager`
- Store users in `/users/<user_id>/`
- Support switching between users
- Add `session_index.json` at the user level

### 🔄 Phase 5: Dual Backend Mode (File or DB)

- Create abstraction layer for `SessionBackend`
  - FlatFileSessionManager
  - DatabaseSessionManager (e.g., SQLAlchemy)
- Toggle backend by config/env
- Write migration tool: flat file → DB

### ✨ Future Extensions

- Session export (zip + manifest)
- Encryption per session or user
- Real-time sync to cloud storage
- GUI session browser and file viewer

---

## 🧠 Design Philosophy

- Each session is **self-contained**: all relevant files are stored together
- Every file is **JSON, JSONL, or NPZ** — portable and easy to convert
- All logic is encapsulated in simple Python modules (e.g. `dy_session_manager.py`)
- File structure mirrors a potential SQL schema for easy transition
- Grows from MVP to full assistant memory system without major rewrites

---

## 🧩 Next Step

Begin with `FlatFileSessionManager` in Phase 1. Once stable, we plug in other modules like persona, file handler, and vector index manager.

