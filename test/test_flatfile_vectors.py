#!/usr/bin/env python3
"""
Test script for vector implementation in flatfile chat database.
This script should be run from the parent directory of flatfile_chat_database.
"""

import asyncio
import json
from pathlib import Path
import shutil
from datetime import datetime
import sys

# Add the parent directory to Python path to import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now import from the package
from flatfile_chat_database import (
    StorageManager, StorageConfig, DocumentRAGPipeline,
    SearchQuery, AdvancedSearchEngine
)


async def test_vector_implementation():
    """Run comprehensive tests of vector implementation."""
    
    print("🚀 Starting Vector Implementation Tests\n")
    
    # Setup test environment
    test_dir = Path("./test_vector_data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Create test configuration
    config = StorageConfig(
        storage_base_path=str(test_dir),
        default_embedding_provider="nomic-ai",  # Local provider, no API needed
        default_chunking_strategy="optimized_summary"
    )
    
    # Initialize storage manager
    storage = StorageManager(config)
    await storage.initialize()
    
    # Test data
    user_id = "test_user"
    session_id = "test_session_001"
    
    print("✅ Storage manager initialized\n")
    
    # ======================
    # Test 1: Basic Storage
    # ======================
    print("📝 Test 1: Basic Document Storage with Vectors")
    
    # Create user and session
    await storage.create_user(user_id)
    await storage.create_session(user_id, session_id)
    
    # Test document
    test_content = """
    Artificial Intelligence (AI) is transforming the world in unprecedented ways. 
    Machine learning algorithms can now understand natural language, recognize images, 
    and even generate creative content. Deep learning neural networks have revolutionized 
    computer vision and natural language processing tasks.
    
    The impact of AI extends across industries - from healthcare where it aids in 
    disease diagnosis, to finance where it detects fraud, to transportation with 
    self-driving vehicles. However, with great power comes great responsibility. 
    Ethical AI development ensures fairness, transparency, and accountability.
    
    Future developments in AI include more advanced reasoning capabilities, better 
    understanding of context and nuance, and improved efficiency in learning from 
    limited data. The goal is to create AI systems that augment human capabilities 
    rather than replace them.
    """
    
    # Store document with vectors
    try:
        success = await storage.store_document_with_vectors(
            user_id=user_id,
            session_id=session_id,
            document_id="test_doc_001",
            content=test_content,
            metadata={"source": "test", "topic": "AI overview"}
        )
        
        print(f"Document stored with vectors: {success}")
        
        # Check vector stats
        stats = await storage.get_vector_stats(user_id, session_id)
        print(f"Vector stats: {json.dumps(stats, indent=2)}\n")
    except Exception as e:
        print(f"⚠️  Note: {e}")
        print("This might be because sentence-transformers is not installed.")
        print("The system will use mock embeddings instead.\n")
    
    # ======================
    # Test 2: Vector Search
    # ======================
    print("🔍 Test 2: Vector Search")
    
    # Test different queries
    test_queries = [
        "machine learning applications",
        "ethical AI development",
        "healthcare diagnosis",
        "neural networks"
    ]
    
    for query in test_queries:
        print(f"\nSearching for: '{query}'")
        try:
            results = await storage.vector_search(
                user_id=user_id,
                query=query,
                session_ids=[session_id],
                top_k=3,
                threshold=0.5
            )
            
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results):
                print(f"  {i+1}. Score: {result.relevance_score:.3f}")
                print(f"     Text: {result.content[:100]}...")
                print(f"     Metadata: {result.metadata}")
        except Exception as e:
            print(f"Search error: {e}")
    
    # ======================
    # Test 3: Multiple Documents
    # ======================
    print("\n📚 Test 3: Multiple Documents")
    
    # Add more documents
    doc_contents = {
        "test_doc_002": """
        Python programming is widely used in data science and machine learning. 
        Libraries like NumPy, Pandas, and Scikit-learn provide powerful tools 
        for data analysis. TensorFlow and PyTorch enable deep learning development.
        """,
        "test_doc_003": """
        Climate change is one of the most pressing challenges of our time. 
        Rising global temperatures, melting ice caps, and extreme weather events 
        threaten ecosystems worldwide. Renewable energy solutions are critical.
        """,
        "test_doc_004": """
        Quantum computing represents a paradigm shift in computational power. 
        Unlike classical bits, quantum bits (qubits) can exist in superposition, 
        enabling exponential speedup for certain problems like cryptography.
        """
    }
    
    for doc_id, content in doc_contents.items():
        try:
            success = await storage.store_document_with_vectors(
                user_id=user_id,
                session_id=session_id,
                document_id=doc_id,
                content=content
            )
            print(f"Stored {doc_id}: {success}")
        except Exception as e:
            print(f"Error storing {doc_id}: {e}")
    
    # Updated stats
    try:
        stats = await storage.get_vector_stats(user_id, session_id)
        print(f"\nUpdated vector stats: {json.dumps(stats, indent=2)}")
    except:
        print("Could not get vector stats")
    
    # ======================
    # Test 4: Hybrid Search
    # ======================
    print("\n🔀 Test 4: Hybrid Search (Text + Vector)")
    
    # First add messages for text search
    from flatfile_chat_database import Message
    
    user_msg = Message(
        role="user",
        content="Tell me about Python libraries for machine learning"
    )
    await storage.add_message(user_id, session_id, user_msg)
    
    assistant_msg = Message(
        role="assistant",
        content="Python has excellent ML libraries including TensorFlow and PyTorch"
    )
    await storage.add_message(user_id, session_id, assistant_msg)
    
    # Hybrid search
    try:
        hybrid_results = await storage.hybrid_search(
            user_id=user_id,
            query="Python machine learning libraries",
            session_ids=[session_id],
            top_k=5,
            vector_weight=0.7  # 70% vector, 30% text
        )
        
        print("\nHybrid search results:")
        for i, result in enumerate(hybrid_results):
            print(f"{i+1}. Type: {result.type}, Score: {result.relevance_score:.3f}")
            print(f"   Content: {result.content[:100]}...")
    except Exception as e:
        print(f"Hybrid search error: {e}")
    
    # ======================
    # Test 5: Document Pipeline
    # ======================
    print("\n⚙️ Test 5: Document Pipeline")
    
    # Create test file
    test_file = test_dir / "test_document.txt"
    test_file.write_text("""
    The future of transportation is electric and autonomous. Electric vehicles (EVs) 
    are becoming mainstream with improved battery technology and charging infrastructure. 
    Autonomous vehicles use AI and sensor fusion to navigate safely without human input.
    """)
    
    # Process through pipeline
    pipeline = DocumentRAGPipeline(config)
    try:
        result = await pipeline.process_document(
            document_path=str(test_file),
            user_id=user_id,
            session_id=session_id,
            document_id="pipeline_doc_001"
        )
        
        print(f"Pipeline processing result:")
        print(f"  Success: {result.success}")
        print(f"  Chunks: {result.chunk_count}")
        print(f"  Vectors: {result.vector_count}")
        print(f"  Time: {result.processing_time:.3f}s")
    except Exception as e:
        print(f"Pipeline error: {e}")
    
    # ======================
    # Test 6: Chunking Strategies
    # ======================
    print("\n📄 Test 6: Different Chunking Strategies")
    
    strategies = ["optimized_summary", "default_fixed", "sentence_short"]
    test_text = """Natural language processing has evolved significantly. 
    Modern NLP models can understand context and generate human-like text. 
    Applications include translation, summarization, and question answering."""
    
    for strategy in strategies:
        try:
            chunks = await storage.chunking_engine.chunk_text(test_text, strategy=strategy)
            print(f"\nStrategy '{strategy}': {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {i+1}: {chunk[:50]}...")
        except Exception as e:
            print(f"Chunking error with {strategy}: {e}")
    
    # ======================
    # Summary
    # ======================
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    try:
        final_stats = await storage.get_vector_stats(user_id, session_id)
        print(f"Total vectors stored: {final_stats.get('total_vectors', 0)}")
        print(f"Total documents: {final_stats.get('total_documents', 0)}")
        print(f"Vector dimensions: {final_stats.get('vector_dimensions', 0)}")
        print(f"Storage size: {final_stats.get('storage_size_bytes', 0)} bytes")
        print(f"Providers used: {final_stats.get('providers_used', [])}")
    except:
        print("Could not get final stats")
    
    print("\n✅ Tests completed!")
    
    # Cleanup
    print("\n🧹 Cleaning up test data...")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    print("Done!")


if __name__ == "__main__":
    print("Flatfile Chat Database - Vector Implementation Test\n")
    print("Note: This test will use mock embeddings if sentence-transformers is not installed.\n")
    
    # Run main tests
    asyncio.run(test_vector_implementation())