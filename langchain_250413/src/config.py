# Configuration settings for the Document QA System
# Document QA System Configuration Settings

# Chunking Configuration
CHUNK_SIZE = 256  # Token size for each chunk
CHUNK_OVERLAP = 26  # 10% overlap (256 * 0.1)

# Vector Store Settings
VECTOR_STORE_PATH = "vector_store"

# Evaluation Parameters
RELEVANCE_THRESHOLD = 0.5  # Minimum similarity score

# CSV file path configuration
csv_path = "chinese_simpleqa.csv"
