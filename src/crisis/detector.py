import re
import numpy as np
from typing import Tuple, Optional, Any, List
import logging
import json
import os
import time
import threading

logger = logging.getLogger(__name__)

CACHE_FILE = os.path.join(os.path.dirname(__file__), "crisis_embeddings_cache.json")

class CrisisDetector:
    def __init__(self, embeddings_model: Optional[Any] = None):
        """
        :param embeddings_model: Object with embed_query(text) -> List[float]
        """
        # Keywords for fast Regex match
        self.crisis_keywords = [
            r"kill myself",
            r"suicid",
            r"end my life",
            r"want to die",
            r"hurt myself",
            r"slit my wrist",
            r"overdose",
            r"no reason to live",
            r"better off dead",
            r"I quit"
        ]
        
        # Semantic Phrases for Embedding Similarity
        self.semantic_phrases = [
            "I don't want to live anymore",
            "The pain is too much to bear, I want out",
            "I'm planning to end it all",
            "Goodbye everyone, I'm done",
            "There is no hope for me",
            "I am going to jump off a bridge",
            "I have a gun and I'm going to use it"
        ]
        
        self.embeddings_model = embeddings_model
        self.phrase_embeddings = []

        if self.embeddings_model:
            # Run initialization in background to avoid blocking server startup
            threading.Thread(target=self._initialize_embeddings, daemon=True).start()

    def _initialize_embeddings(self):
        """Attempts to load embeddings from cache, or generates them using local model."""
        logger.info("CrisisDetector: Initializing embeddings in background...")
        
        # 1. Try Loading from Cache
        if self._load_from_cache():
            return

        # 2. Generate (Local)
        embeddings = self._generate_embeddings()
        
        # 3. Save to Cache if successful
        if embeddings:
            self.phrase_embeddings = embeddings
            self._save_to_cache()
        else:
            logger.warning("⚠️ CrisisDetector: Semantic search disabled due to embedding failure.")

    def _load_from_cache(self) -> bool:
        """Returns True if valid embeddings were loaded from disk."""
        if not os.path.exists(CACHE_FILE):
            return False
            
        try:
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
            
            # Verify cache validity (must match current phrases)
            if data.get("phrases") == self.semantic_phrases:
                self.phrase_embeddings = data["embeddings"]
                logger.info(f"✅ CrisisDetector: Loaded {len(self.phrase_embeddings)} embeddings from cache.")
                return True
            else:
                logger.info("CrisisDetector: Cache stale (phrases changed). Recomputing...")
        except Exception as e:
            logger.warning(f"CrisisDetector: Failed to read cache: {e}")
            
        return False

    def _save_to_cache(self):
        """Saves current embeddings to disk."""
        try:
            data = {
                "phrases": self.semantic_phrases,
                "embeddings": self.phrase_embeddings
            }
            with open(CACHE_FILE, 'w') as f:
                json.dump(data, f)
            logger.info(f"💾 CrisisDetector: Saved embeddings to {CACHE_FILE}")
        except Exception as e:
            logger.error(f"CrisisDetector: Failed to save cache: {e}")

    def _generate_embeddings(self) -> List[List[float]]:
        """Generates embeddings using the local model."""
        try:
            logger.info("CrisisDetector: Generating embeddings locally...")
            if hasattr(self.embeddings_model, "embed_documents"):
                return self.embeddings_model.embed_documents(self.semantic_phrases)
            else:
                return [self.embeddings_model.embed_query(p) for p in self.semantic_phrases]
        except Exception as e:
            logger.error(f"CrisisDetector: Embedding generation failed: {e}")
            return []

    def _chunk_text(self, text: str, window_size: int = 15, step: int = 10) -> List[str]:
        """Splits text into overlapping chunks of words."""
        words = text.split()
        if len(words) <= window_size:
            return [text]
        
        chunks = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + window_size])
            chunks.append(chunk)
            if i + window_size >= len(words):
                break
        return chunks

    def detect(self, text: str) -> Tuple[bool, str]:
        """
        Analyzes text for crisis content using Regex OR (Semantic Search with Sliding Window).
        :param text: User input text
        :return: (is_crisis, reason)
        """
        # 1. Regex Check
        text_lower = text.lower()
        for pattern in self.crisis_keywords:
            if re.search(pattern, text_lower):
                return True, "User is expressing suicidal thoughts or self-harm intent"

        # 2. Semantic Search Check
        if self.embeddings_model and self.phrase_embeddings:
            try:
                # Chunk the text to catch phrases hidden in long messages
                chunks = self._chunk_text(text)
                max_score_overall = 0
                best_phrase = ""
                
                for chunk in chunks:
                    user_vector = self.embeddings_model.embed_query(chunk)
                    threshold = 0.85 
                    
                    user_norm = np.linalg.norm(user_vector)
                    if user_norm == 0: continue
                    
                    for phrase_vec, phrase_text in zip(self.phrase_embeddings, self.semantic_phrases):
                        phrase_norm = np.linalg.norm(phrase_vec)
                        if phrase_norm == 0: continue
                        
                        score = np.dot(user_vector, phrase_vec) / (user_norm * phrase_norm)
                        if score > max_score_overall:
                            max_score_overall = score
                            best_phrase = phrase_text
                
                logger.info(f"Crisis Check: Max Score {max_score_overall:.4f} (Matched: '{best_phrase}')")

                if max_score_overall > threshold:
                    return True, "User content indicates severe distress or crisis intent"
                    
            except Exception as e:
                logger.error(f"Semantic Check Failed: {e}")
                
        
        return False, ""
