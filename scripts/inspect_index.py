"""Inspect the baked vector index.

Prints chunk counts per source, dumps the actual text of the concept-page chunks
(to check whether real content was ingested or just boilerplate), and shows what
a query retrieves.

Usage (the index must already be built):
    python scripts/inspect_index.py ["your query"]
"""

import collections
import os
import sys
import warnings

warnings.filterwarnings("ignore")  # silence the relevance-score range warning

# Make `import src` work however this script is launched.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag import load_vectorstore, retrieve  # noqa: E402

vs = load_vectorstore()

# 1. What is actually in the index?
data = vs._collection.get(include=["metadatas", "documents"])
metas, texts = data["metadatas"], data["documents"]
counts = collections.Counter((m or {}).get("source", "?") for m in metas)
print(f"total chunks: {sum(counts.values())}\n")
print("chunks per source:")
for src, n in counts.most_common():
    print(f"  {n:4d}  {src}")

# 2. Dump the real content of the concept pages (boilerplate vs real prose?).
for needle in ("retrieval", "knowledge-base"):
    print(f"\n===== chunks from {needle} =====")
    i = 0
    for meta, text in zip(metas, texts):
        if needle in (meta or {}).get("source", ""):
            i += 1
            print(f"\n--- chunk {i} ({len(text)} chars) ---")
            print(text[:500])

# 3. What the app feeds the model for the query.
query = (
    sys.argv[1]
    if len(sys.argv) > 1
    else ("What's the difference between a retriever and a vectorstore?")
)
print(f"\n===== diversified retrieve for: {query!r} =====")
for doc, score in retrieve(vs, query):
    print(f"  {score:.3f}  {doc.metadata.get('source', '?')}")
