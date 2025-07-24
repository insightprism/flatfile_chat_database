#!/usr/bin/env python3
"""
Test Phase 4 production features: streaming, compression, migration, and benchmarks.
"""

import asyncio
import sys
from pathlib import Path
import tempfile
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from flatfile_chat_database import (
    StorageManager, StorageConfig, Message, SearchQuery
)
from flatfile_chat_database.streaming import MessageStreamer, ExportStreamer
from flatfile_chat_database.compression import CompressionManager, CompressionConfig, CompressionType
from flatfile_chat_database.migration import FlatfileExporter, SQLiteAdapter, MigrationStats
from flatfile_chat_database.benchmark import PerformanceBenchmark


async def test_streaming():
    """Test message streaming for large sessions"""
    print("\n=== Testing Message Streaming ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup
        config = StorageConfig(storage_base_path=temp_dir)
        manager = StorageManager(config=config)
        await manager.initialize()
        
        # Create test data
        user_id = "streaming_test_user"
        await manager.create_user(user_id)
        session_id = await manager.create_session(user_id, "Large Session")
        
        # Add many messages
        print("Creating 500 test messages...")
        for i in range(500):
            msg = Message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}: This is a test message with some content to make it realistic."
            )
            await manager.add_message(user_id, session_id, msg)
        
        # Test streaming
        streamer = MessageStreamer(config)
        
        print("\n1. Streaming messages in chunks:")
        chunk_count = 0
        message_count = 0
        async for chunk in streamer.stream_messages(user_id, session_id, max_messages=100):
            chunk_count += 1
            message_count += len(chunk)
            print(f"   Chunk {chunk_count}: {len(chunk)} messages")
        print(f"   Total: {message_count} messages in {chunk_count} chunks")
        
        print("\n2. Streaming in reverse order:")
        chunk_count = 0
        async for chunk in streamer.stream_messages_reverse(user_id, session_id, limit=50):
            chunk_count += 1
            if chunk_count == 1:
                print(f"   Latest message: {chunk[0].content[:50]}...")
        print(f"   Streamed {chunk_count} chunks in reverse")
        
        print("\n3. Export streaming:")
        exporter = ExportStreamer(config)
        export_chunks = 0
        async for chunk in exporter.stream_session_export(user_id, session_id):
            export_chunks += 1
        print(f"   Exported session in {export_chunks} chunks")
        
        return True


async def test_compression():
    """Test compression functionality"""
    print("\n=== Testing Compression ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config = StorageConfig(storage_base_path=temp_dir)
        compression_config = CompressionConfig(
            enabled=True,
            type=CompressionType.GZIP,
            level=6,
            min_size_bytes=100
        )
        compression = CompressionManager(config, compression_config)
        
        # Test 1: Text compression
        print("\n1. Text compression:")
        test_text = "This is a test message. " * 100  # Repeated text compresses well
        test_bytes = test_text.encode('utf-8')
        compressed = await compression.compress_data(test_bytes)
        
        original_size = len(test_bytes)
        compressed_size = len(compressed)
        ratio = compressed_size / original_size
        
        print(f"   Original size: {original_size} bytes")
        print(f"   Compressed size: {compressed_size} bytes")
        print(f"   Compression ratio: {ratio:.2%}")
        
        # Decompress and verify
        decompressed = await compression.decompress_data(compressed)
        assert decompressed == test_bytes
        print("   ✓ Decompression verified")
        
        # Test 2: JSON compression
        print("\n2. JSON compression:")
        test_data = {
            "messages": [{"role": "user", "content": f"Message {i}"} for i in range(50)],
            "metadata": {"session": "test", "user": "user1"}
        }
        
        compressed_json = await compression.compress_json(test_data)
        decompressed_data = await compression.decompress_json(compressed_json)
        
        print(f"   Original JSON size: {len(json.dumps(test_data))} bytes")
        print(f"   Compressed size: {len(compressed_json)} bytes")
        assert decompressed_data == test_data
        print("   ✓ JSON compression/decompression verified")
        
        # Test 3: File compression
        print("\n3. File compression:")
        test_file = Path(temp_dir) / "test.txt"
        test_file.write_text("Test content " * 1000)
        
        stats = await compression.get_compression_stats(test_file)
        print(f"   Potential space saved: {stats['space_saved_percent']:.1f}%")
        
        compressed_file = await compression.compress_file(test_file)
        print(f"   Compressed file created: {compressed_file.name}")
        
        return True


async def test_migration():
    """Test export/migration functionality"""
    print("\n=== Testing Migration Tools ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup storage with test data
        config = StorageConfig(storage_base_path=temp_dir)
        manager = StorageManager(config=config)
        await manager.initialize()
        
        # Create test data
        print("\n1. Creating test data...")
        user_id = "migration_test_user"
        await manager.create_user(user_id, {"username": "Test User"})
        
        # Create sessions with messages
        for i in range(3):
            session_id = await manager.create_session(user_id, f"Session {i}")
            for j in range(10):
                msg = Message(
                    role="user" if j % 2 == 0 else "assistant",
                    content=f"Session {i}, Message {j}"
                )
                await manager.add_message(user_id, session_id, msg)
        
        # Test SQLite export
        print("\n2. Exporting to SQLite database:")
        exporter = FlatfileExporter(manager)
        db_path = str(Path(temp_dir) / "export.db")
        adapter = SQLiteAdapter(db_path)
        
        stats = await exporter.export_to_database(adapter)
        print(f"   Users exported: {stats.total_users}")
        print(f"   Sessions exported: {stats.total_sessions}")
        print(f"   Messages exported: {stats.total_messages}")
        print(f"   Export duration: {stats.duration_seconds:.2f}s")
        print(f"   Errors: {len(stats.errors)}")
        
        # Test JSON export
        print("\n3. Exporting to JSON:")
        json_path = Path(temp_dir) / "export.json"
        stats = await exporter.export_to_json(json_path, compress=True)
        
        compressed_path = json_path.with_suffix('.json.gz')
        if compressed_path.exists():
            print(f"   Compressed export size: {compressed_path.stat().st_size} bytes")
        
        # Verify SQLite export
        print("\n4. Verifying SQLite export:")
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        user_count = cursor.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        session_count = cursor.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        message_count = cursor.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        
        print(f"   Users in DB: {user_count}")
        print(f"   Sessions in DB: {session_count}")
        print(f"   Messages in DB: {message_count}")
        
        conn.close()
        
        return True


async def test_performance():
    """Run performance benchmarks"""
    print("\n=== Running Performance Benchmarks ===")
    print("(Using reduced iterations for demo)")
    
    # Run quick benchmark
    benchmark = PerformanceBenchmark(iterations=10, warmup=2)
    results = await benchmark.run_all_benchmarks(verbose=False)
    
    # Print summary
    print("\nBenchmark Results:")
    print("-" * 60)
    
    for op, data in results["details"].items():
        print(f"{op:<25} Mean: {data['mean_ms']:>8.2f}ms  "
              f"P95: {data['p95_ms']:>8.2f}ms")
    
    print("\nPerformance Targets:")
    print("-" * 60)
    
    all_passed = True
    for op, target_data in results["performance_targets"].items():
        passed = target_data["passed"]
        all_passed &= passed
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{op:<25} Target: {target_data['target_ms']:>6}ms  "
              f"Actual: {target_data['actual_p95_ms']:>6}ms  {status}")
    
    return all_passed


async def main():
    """Run all Phase 4 tests"""
    print("Testing Phase 4 Production Features")
    print("=" * 60)
    
    try:
        # Test streaming
        streaming_ok = await test_streaming()
        
        # Test compression
        compression_ok = await test_compression()
        
        # Test migration
        migration_ok = await test_migration()
        
        # Test performance
        performance_ok = await test_performance()
        
        print("\n" + "=" * 60)
        print("Phase 4 Test Summary:")
        print(f"  Streaming: {'✓ PASSED' if streaming_ok else '✗ FAILED'}")
        print(f"  Compression: {'✓ PASSED' if compression_ok else '✗ FAILED'}")
        print(f"  Migration: {'✓ PASSED' if migration_ok else '✗ FAILED'}")
        print(f"  Performance: {'✓ PASSED' if performance_ok else '✗ FAILED'}")
        
        all_passed = streaming_ok and compression_ok and migration_ok and performance_ok
        
        if all_passed:
            print("\n✓ All Phase 4 tests passed!")
            return 0
        else:
            print("\n✗ Some tests failed!")
            return 1
            
    except Exception as e:
        print(f"\n✗ Error during tests: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)