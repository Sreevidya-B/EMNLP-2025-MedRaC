# rag.py
"""
Dependencies
------------
pip install langchain faiss-cpu openai tiktoken
# or   pip install faiss-gpu   if you have FAISS with CUDA

Environment
-----------
export OPENAI_API_KEY="your key"
"""

from __future__ import annotations

import os,json
import time
from typing import List, Tuple

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
        # load_dotenv()

class RAG:
    """A minimal RAG class that retrieves the *k* most similar formula blocks."""

    def __init__(
        self,
        doc_path: str = "data/web_formula.txt",
        embedding_model: str = "text-embedding-ada-002", # text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
        embeddings_dir: str = "data/one_shot_finalized_explanation_formulas_embeddings",
        normalize_embeddings: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        doc_path : str
            Path to the plain-text knowledge base. Each document is the text
            between <<FORMULA START>> and <<FORMULA END>> markers.
        embedding_model : str | None
            OpenAI embedding model name. If None, uses OpenAI's default.
        normalize_embeddings : bool
            Whether to L2-normalize vectors before similarity search (recommended).
        """
        # load_dotenv()
        load_dotenv(dotenv_path='/work/pi_hongyu_umass_edu/sreevidyabol_umass_edu/MedRaC/EMNLP-2025-MedRaC/.env')
        
        self.api_key = os.getenv("OPENAI_API_KEY")

        # Try loading a precomputed FAISS index
        if os.path.isdir(embeddings_dir):
            embeddings = OpenAIEmbeddings(
                model=embedding_model,
                openai_api_key=self.api_key
            )
            # <-- Allow your own index to deserialize
            self.vectorstore = FAISS.load_local(
                embeddings_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
            return

        # Otherwise, load & split the source file
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"Knowledge base not found: {doc_path}")

        # 1. Load the entire file and split by special markers -------------
        with open(doc_path, "r", encoding="utf-8") as fh:
            text = fh.read()

        self.documents: List[Document] = []
        start_marker = "<<FORMULA START>>"
        end_marker = "<<FORMULA END>>"
        pos = 0
        block_idx = 0

        while True:
            start = text.find(start_marker, pos)
            if start == -1:
                break
            start += len(start_marker)
            end = text.find(end_marker, start)
            if end == -1:
                break

            # Extract and clean the block content
            content = text[start:end].strip()
            if content:
                self.documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "block_index": block_idx,
                            "source_file": doc_path,
                        },
                    )
                )
                block_idx += 1

            # Move past this end marker for the next search
            pos = end + len(end_marker)

        if not self.documents:
            raise ValueError(
                "No documents found between <<FORMULA START>> and <<FORMULA END>> markers."
            )

        # 2. Create embeddings & vector store -------------------------------
        embeddings = OpenAIEmbeddings(model=embedding_model,base_url="https://api.openai.com/v1",openai_api_key=self.api_key)
        # Build an in‑memory FAISS index
        self.vectorstore = FAISS.from_documents(
            self.documents,
            embeddings,
            normalize_L2=normalize_embeddings,
            # batch_size=100  
        )

        os.makedirs(embeddings_dir, exist_ok=True)
        self.vectorstore.save_local(embeddings_dir)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 1) -> List[Tuple[str, float]]:
        """
        Return up to k most similar formula blocks for the given query.

        Parameters
        ----------
        query : str
            Arbitrary text used as retrieval key.
        k : int, default 1
            Number of top results wanted.

        Returns
        -------
        List[Tuple[str, float]]
            Each tuple is (block_text, similarity_score). Higher score == closer.
        """
        
        if k < 1:
            raise ValueError("k must be >= 1")
        query = self.trim_before_phrase(query)
        # print(f"Query: {query}")
        # similarity_search_with_score returns List[Tuple[Document, score]]
        hits = self.vectorstore.similarity_search_with_relevance_scores(query, k=k)
        
        # Convert to (content, score) pairs. LangChain returns higher == more similar.
        return [(doc.page_content, float(score)) for doc, score in hits]

    def __len__(self) -> int:
        """Return the number of indexed formula blocks."""
        return int(self.vectorstore.index.ntotal)

    @staticmethod
    def evaluate_rag_on_formula_accuracy(json_path: str,
                                        doc_path: str,
                                        top_k: int = 1):
        """
        Evaluate the accuracy of formula retrieval using a RAG retriever.

        Args:
            json_path: Path to the JSON file containing questions and true formulas.
            doc_path: Path to the document used to build the RAG index.
            top_k: Number of top retrieval results to consider for each question.
        """

        # 1. Load the records to evaluate
        with open(json_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        # 2. Initialize the RAG retriever
        rag = RAG(doc_path=doc_path, embedding_model="text-embedding-3-small")

        correct_count = 0
        total = len(records)

        # 3. Iterate over each record and perform retrieval and comparison
        for entry in records:
            calc_id      = entry.get("Calculator ID", "<unknown>")
            question     = entry["Question"]
            true_formula = entry["Formula"]

            # Retrieve the top_k most similar passages
            results = rag.retrieve(question, k=top_k)

            if not results:
                print(f"[{calc_id}] No retrieval results; marking as incorrect.")
                continue

            # --- 修改开始 ----------
            match_found = False
            for idx, (retrieved_text, score) in enumerate(results, start=1):
                if retrieved_text.strip() == true_formula.strip():
                    match_found = True
                    if top_k > 1:
                        print(f"[{calc_id}] Match found at rank {idx} (score={score:.4f})")
                    break
            # --- 修改结束 ----------

            if match_found:
                correct_count += 1
                status = "Correct"
            else:
                status = "Incorrect"
                print(f"\n[Calculator ID: {calc_id}] Comparison failed:")
                print(f"Question: {question}")
                for idx, (retrieved_text, score) in enumerate(results, start=1):
                    print(f"Retrieved #{idx}: {retrieved_text} (score = {score})")
                print("Expected formula:")
                print(true_formula)
            print(f"[{calc_id}] Determination: {status}")

        # 4. Compute and print overall accuracy
        accuracy = (correct_count / total * 100) if total > 0 else 0.0
        print("\n====== Summary ======")
        print(f"Total records: {total}")
        print(f"Correct count: {correct_count}")
        print(f"Accuracy: {accuracy:.2f}%")

    @staticmethod
    def trim_before_phrase(text: str) -> str:
        """
        If the phrase "You should use the patient's medical values" appears in the text,
        return only the portion before that phrase. Otherwise, return the original text.
        """
        phrase = "You should use the patient's medical values"
        idx = text.find(phrase)
        if idx != -1:
            return text[:idx]
        return text

# -------------------------------------------------------------------------
# Usage example (uncomment to run)
# -------------------------------------------------------------------------
# if __name__ == "__main__":
#     rag = RAG(doc_path="formula.txt")
#     question = "How do I compute Cockcroft-Gault clearance?"
#     top_k = 5
#     results = rag.retrieve(question, k=top_k)
#     print(f"\nTop {len(results)} formula blocks for: “{question}”\n" + "-" * 60)
#     for rank, (text, score) in enumerate(results, start=1):
#         print(f"{rank:>2}. ({score:.4f})  {text}\n")

# -------------------------------------------------------------------------
# Usage example (uncomment to run)
# -------------------------------------------------------------------------
# if __name__ == "__main__":
#     rag = RAG(doc_path="/work/pi_hongyu_umass_edu/sreevidyabol_umass_edu/MedRaC/EMNLP-2025-MedRaC/data/web_formula.txt")
#     question = "Based on the patient's last menstrual period, what is the patient's estimated gestational age? Your answer should be a tuple, specifying the number of weeks and days (i.e. (4 weeks, 3 days), (0 weeks, 5 days), (1 week, 5 days), (8 weeks, 0 days))."
#     top_k = 5
#     results = rag.retrieve(question, k=top_k)
#     print(f"\nTop {len(results)} formula blocks for: “{question}”\n" + "-" * 60)
#     for rank, (text, score) in enumerate(results, start=1):
#         print(f"{rank:>2}. ({score:.4f})  {text}\n")
