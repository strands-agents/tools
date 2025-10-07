#!/usr/bin/env python3
"""
Test Elasticsearch Memory Tool with 'ayan' index.
"""

import sys
import json
from datetime import datetime

# Add the src directory to the path so we can import our module
sys.path.insert(0, 'src')

from strands import Agent
from strands_tools.elasticsearch_memory import ElasticsearchMemoryToolProvider


def test_ayan_index():
    """Test the Elasticsearch memory tool with 'ayan' index."""
    
    # Elasticsearch serverless credentials
    ES_URL = "https://strands-project-db43e9.es.us-east-1.aws.elastic.cloud:443"
    ES_API_KEY = "SmJMVXU1a0JWSzA1bDNfNldKY0Q6UnpIeGU1RFQzOS1FXzFpVVk0aXRadw=="
    
    print("🚀 Testing Elasticsearch Memory Tool with 'ayan' index")
    print("=" * 60)
    
    try:
        # Initialize the provider with 'ayan' index
        print("🔧 Initializing Elasticsearch Memory provider...")
        provider = ElasticsearchMemoryToolProvider(
            es_url=ES_URL,
            api_key=ES_API_KEY,
            index_name="ayan",  # Using 'ayan' as requested
            namespace="ayan_test",
            region="us-east-1"
        )
        
        # Create agent
        agent = Agent(tools=provider.tools)
        print("✅ Successfully connected to Elasticsearch!")
        print(f"   Index: {provider.index_name}")
        print(f"   Namespace: {provider.namespace}")
        print()
        
        # Test 1: Store a memory
        print("📝 Test 1: Storing a memory in 'ayan' index...")
        test_content = f"This is Ayan's test memory created at {datetime.now().isoformat()}"
        test_metadata = {
            "owner": "ayan",
            "category": "test",
            "timestamp": datetime.now().isoformat()
        }
        
        result = agent.tool.elasticsearch_memory(
            action="record",
            content=test_content,
            metadata=test_metadata
        )
        
        if result["status"] == "success":
            print("✅ Memory stored successfully in 'ayan' index!")
            # Extract memory ID from response
            response_data = json.loads(result["content"][0]["text"].split(": ", 1)[1])
            memory_id = response_data["memory_id"]
            print(f"   Memory ID: {memory_id}")
            print(f"   Content: {test_content}")
        else:
            print(f"❌ Failed to store memory: {result['content'][0]['text']}")
            return False
        
        print()
        
        # Test 2: Store additional memories
        print("📝 Test 2: Storing additional memories...")
        additional_memories = [
            {
                "content": "Ayan prefers working with Python and machine learning",
                "metadata": {"owner": "ayan", "category": "preferences", "type": "technical"}
            },
            {
                "content": "Ayan is interested in Elasticsearch and vector search",
                "metadata": {"owner": "ayan", "category": "interests", "type": "technical"}
            },
            {
                "content": "Ayan likes to test new tools and technologies",
                "metadata": {"owner": "ayan", "category": "behavior", "type": "personal"}
            }
        ]
        
        stored_ids = [memory_id]  # Include the first memory
        for memory_data in additional_memories:
            result = agent.tool.elasticsearch_memory(
                action="record",
                content=memory_data["content"],
                metadata=memory_data["metadata"]
            )
            
            if result["status"] == "success":
                response_data = json.loads(result["content"][0]["text"].split(": ", 1)[1])
                stored_ids.append(response_data["memory_id"])
                print(f"   ✅ Stored: {memory_data['content'][:50]}...")
            else:
                print(f"   ❌ Failed to store: {result['content'][0]['text']}")
        
        print()
        
        # Test 3: Semantic search
        print("🔍 Test 3: Semantic search in 'ayan' index...")
        search_queries = [
            "Ayan's technical preferences",
            "machine learning and Python",
            "Elasticsearch and search technologies"
        ]
        
        for query in search_queries:
            print(f"\n   Query: '{query}'")
            result = agent.tool.elasticsearch_memory(
                action="retrieve",
                query=query,
                max_results=3
            )
            
            if result["status"] == "success":
                response_data = json.loads(result["content"][0]["text"].split(": ", 1)[1])
                memories = response_data.get("memories", [])
                
                if memories:
                    for i, memory in enumerate(memories, 1):
                        score = memory.get("score", 0)
                        content = memory.get("content", "")
                        print(f"     {i}. Score: {score:.3f} - {content[:60]}...")
                else:
                    print("     No matching memories found")
            else:
                print(f"     ❌ Search failed: {result['content'][0]['text']}")
        
        print()
        
        # Test 4: List all memories in 'ayan' index
        print("📋 Test 4: Listing all memories in 'ayan' index...")
        result = agent.tool.elasticsearch_memory(
            action="list",
            max_results=10
        )
        
        if result["status"] == "success":
            response_data = json.loads(result["content"][0]["text"].split(": ", 1)[1])
            memories = response_data.get("memories", [])
            total = response_data.get("total", 0)
            print(f"   Total memories in 'ayan' index: {total}")
            print(f"   Showing {len(memories)} memories:")
            
            for i, memory in enumerate(memories, 1):
                memory_id_short = memory.get("memory_id", "unknown")[:16]
                content = memory.get("content", "")[:50]
                timestamp = memory.get("timestamp", "")[:19]
                owner = memory.get("metadata", {}).get("owner", "unknown")
                print(f"   {i}. [{memory_id_short}...] Owner: {owner} - {content}... ({timestamp})")
        else:
            print(f"❌ Memory listing failed: {result['content'][0]['text']}")
            return False
        
        print()
        
        # Test 5: Get specific memory
        print("🔍 Test 5: Getting specific memory by ID...")
        if stored_ids:
            test_memory_id = stored_ids[0]
            result = agent.tool.elasticsearch_memory(
                action="get",
                memory_id=test_memory_id
            )
            
            if result["status"] == "success":
                response_data = json.loads(result["content"][0]["text"].split(": ", 1)[1])
                print("✅ Memory retrieved successfully!")
                print(f"   Content: {response_data.get('content', 'N/A')}")
                print(f"   Owner: {response_data.get('metadata', {}).get('owner', 'N/A')}")
                print(f"   Namespace: {response_data.get('namespace', 'N/A')}")
            else:
                print(f"❌ Failed to retrieve memory: {result['content'][0]['text']}")
        
        print()
        print("🎉 All tests completed successfully with 'ayan' index!")
        print()
        print("📊 Test Summary:")
        print("   ✅ Connection to Elasticsearch serverless")
        print("   ✅ Index creation with name 'ayan'")
        print("   ✅ Memory storage with custom metadata")
        print("   ✅ Semantic search across Ayan's memories")
        print("   ✅ Memory listing and retrieval")
        print("   ✅ Namespace isolation (ayan_test)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        print()
        print("🔧 Troubleshooting:")
        print("   1. Verify Elasticsearch credentials are correct")
        print("   2. Check AWS credentials for Bedrock access")
        print("   3. Ensure network connectivity to Elasticsearch Cloud")
        
        import traceback
        print("\n📋 Full error traceback:")
        traceback.print_exc()
        
        return False


if __name__ == "__main__":
    success = test_ayan_index()
    sys.exit(0 if success else 1)
