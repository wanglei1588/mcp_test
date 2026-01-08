import chromadb


def list_collections(db_path):
    client = chromadb.PersistentClient(db_path)
    collections = client.list_collections()
    print("len:", len(collections))

    for i, collection in enumerate(collections):
        print(f"{i+1}. {collection.name}, {collection.count()}")


def delete_collection(db_path, collection_name):
    try:
        client = chromadb.PersistentClient(db_path)
        client.delete_collection(collection_name)
        print(f"Collection '{collection_name}' deleted successfully.")
    except Exception as e:
        print(f"Error deleting collection: {e}")


if __name__ == '__main__':
    list_collections("./asset/chroma-1")
