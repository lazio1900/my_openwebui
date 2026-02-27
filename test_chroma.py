import chromadb
try:
    c = chromadb.HttpClient(host="localhost", port=8000)
    print("heartbeat:", c.heartbeat())
    print("collections:", c.list_collections())
except Exception as e:
    print(f"ERROR: {e}")
