#!/usr/bin/env python3
"""
Example usage of vector functionality in flatfile chat database.

This demonstrates how to:
1. Store documents with automatic vectorization
2. Perform vector similarity search
3. Use hybrid search (text + vector)
4. Process documents through the pipeline
"""

import asyncio
from pathlib import Path
import sys

# Add parent directory to path if running from this directory
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).parent.parent))

from flatfile_chat_database import (
    StorageManager, 
    StorageConfig,
    DocumentRAGPipeline
)


async def main():
    """Demonstrate vector functionality."""
    
    # 1. Initialize with default configuration
    # Uses Nomic-AI (local) and optimized_summary chunking by default
    config = StorageConfig(storage_base_path="./my_chat_data")
    storage = StorageManager(config)
    await storage.initialize()
    
    # Create user and session
    user_id = "demo_user"
    session_id = "demo_session"
    await storage.create_user(user_id)
    await storage.create_session(user_id, session_id)
    
    print("🚀 Flatfile Vector Database Demo\n")
    
    # 2. Store a document with automatic vectorization
    print("📄 Storing document with vectors...")
    
    document_content = """
    Renewable energy sources like solar, wind, and hydroelectric power are 
    becoming increasingly important in the fight against climate change. 
    Solar panels convert sunlight directly into electricity using photovoltaic cells.
    Wind turbines harness kinetic energy from moving air to generate power.
    Hydroelectric dams use the force of flowing water to turn turbines.
    
    The transition to renewable energy requires significant infrastructure 
    investment but offers long-term benefits including reduced carbon emissions,
    energy independence, and sustainable economic growth.
    """
    
    success = await storage.store_document_with_vectors(
        user_id=user_id,
        session_id=session_id,
        document_id="renewable_energy_doc",
        content=document_content,
        metadata={"topic": "renewable energy", "type": "overview"}
    )
    
    print(f"✅ Document stored: {success}")
    
    # Check what was stored
    stats = await storage.get_vector_stats(user_id, session_id)
    print(f"📊 Stored {stats['total_vectors']} vectors for {stats['total_documents']} documents\n")
    
    # 3. Perform vector search
    print("🔍 Searching for similar content...")
    
    queries = [
        "solar panel technology",
        "environmental benefits of clean energy",
        "water power generation"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        results = await storage.vector_search(
            user_id=user_id,
            query=query,
            top_k=2,
            threshold=0.5
        )
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result.relevance_score:.3f}")
            print(f"     Text: {result.content[:80]}...")
    
    # 4. Add more content for hybrid search
    print("\n\n📝 Adding more documents...")
    
    # Add another document
    await storage.store_document_with_vectors(
        user_id=user_id,
        session_id=session_id,
        document_id="solar_tech_doc",
        content="""
        Recent advances in solar technology include perovskite solar cells,
        which promise higher efficiency at lower costs. Concentrated solar power
        uses mirrors to focus sunlight for thermal energy generation.
        """
    )
    
    # 5. Demonstrate hybrid search
    print("\n🔀 Hybrid Search (combining text and vector search)")
    
    # Add some chat messages for text search
    from flatfile_chat_database import Message
    
    messages = [
        Message(role="user", content="What are the latest solar innovations?"),
        Message(role="assistant", content="Recent solar innovations include perovskite cells and concentrated solar power systems.")
    ]
    
    for msg in messages:
        await storage.add_message(user_id, session_id, msg)
    
    # Perform hybrid search
    hybrid_results = await storage.hybrid_search(
        user_id=user_id,
        query="solar technology innovations",
        top_k=3,
        vector_weight=0.7  # 70% vector, 30% text matching
    )
    
    print("\nHybrid search results:")
    for i, result in enumerate(hybrid_results, 1):
        print(f"{i}. Type: {result.type}, Score: {result.relevance_score:.3f}")
        print(f"   Content: {result.content[:100]}...")
    
    # 6. Use the document pipeline
    print("\n\n⚙️  Document Pipeline Example")
    
    # Create a test file
    test_file = Path("./example_doc.txt")
    test_file.write_text("""
    Artificial intelligence is revolutionizing healthcare through:
    - Diagnostic imaging analysis
    - Drug discovery acceleration  
    - Personalized treatment plans
    - Predictive health monitoring
    """)
    
    pipeline = DocumentRAGPipeline(config)
    result = await pipeline.process_document(
        document_path=str(test_file),
        user_id=user_id,
        session_id=session_id,
        document_id="ai_healthcare_doc"
    )
    
    print(f"Pipeline result: Success={result.success}, Chunks={result.chunk_count}, Time={result.processing_time:.2f}s")
    
    # Clean up
    test_file.unlink()
    
    # 7. Final statistics
    print("\n📈 Final Statistics")
    final_stats = await storage.get_vector_stats(user_id, session_id)
    print(f"Total vectors: {final_stats['total_vectors']}")
    print(f"Total documents: {final_stats['total_documents']}")
    print(f"Storage size: {final_stats['storage_size_bytes']:,} bytes")
    
    print("\n✅ Demo complete! Check './my_chat_data' to see the stored files.")


if __name__ == "__main__":
    asyncio.run(main())