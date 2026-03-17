"""
torchserve/handler.py
---------------------
TorchServe model handler for ONNX-based sentiment inference.

ARCHITECTURE NOTE — why this handler exists:
    TorchServe expects a handler that receives raw HTTP request data and
    returns a prediction.  Because ONNX only captured the neural network
    (not the full sklearn pipeline), this handler must replicate the two
    preprocessing steps that happen *before* the ONNX model:

        raw text (string)
          └─> TextCleaner  (regex, reimplemented inline — no sklearn import needed)
                └─> TF-IDF vectorizer (.pkl loaded at initialise time)
                      └─> dense float32 tensor
                            └─> ONNX InferenceSession
                                  └─> softmax → class + confidence

What it does:
    Implements the four-method TorchServe handler lifecycle:
        initialize()   — load ONNX model + TF-IDF vectorizer once at startup
        preprocess()   — clean text, vectorize, convert to float32 ndarray
        inference()    — run ONNX session, apply softmax
        postprocess()  — format output as structured JSON

Parameters / configuration:
    All file paths are resolved relative to the model_dir passed by TorchServe.
    Expected files inside the .mar archive:
        onnx_model.onnx
        tfidf_vectorizer.pkl

Returns:
    List[dict]: One dict per input item:
        {
            "prediction":  "positive" | "negative",
            "confidence":  float (0.0 – 1.0),
            "class_scores": {"negative": float, "positive": float}
        }

Connects to:
    export_onnx.py  — produces onnx_model.onnx + tfidf_vectorizer.pkl
    serve.sh        — packages this handler into the .mar archive
"""

import json
import logging
import os
import re
from typing import Any, Dict, List

import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Labels must match the integer encoding used during training
# (0 = negative, 1 = positive — see prepare_data_csv.py)
LABEL_MAP: Dict[int, str] = {0: "negative", 1: "positive"}


class SentimentHandler:
    """
    TorchServe BaseHandler-compatible handler for sentiment classification.

    Implements the four-phase TorchServe lifecycle without subclassing
    BaseHandler so it can also be unit-tested directly without a running
    TorchServe instance.
    """

    def __init__(self) -> None:
        self.onnx_session = None
        self.vectorizer   = None
        self.initialized  = False

    # ------------------------------------------------------------------
    # Phase 1 of TorchServe lifecycle
    # ------------------------------------------------------------------

    def initialize(self, context) -> None:
        """
        Load the ONNX model and TF-IDF vectorizer at worker startup.

        What it does:
            Called once when TorchServe loads the model archive.  Reads
            model_dir from the TorchServe context and loads both artefacts
            from that directory.

        Parameters:
            context: TorchServe context object with .system_properties dict
                     containing 'model_dir'.

        Returns:
            None

        Raises:
            FileNotFoundError: If onnx_model.onnx or tfidf_vectorizer.pkl
                               are missing from model_dir.
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for the TorchServe handler. "
                "Install with: pip install onnxruntime"
            )

        model_dir = context.system_properties.get("model_dir")
        logger.info(f"Initialising handler from model_dir='{model_dir}'")

        onnx_path = os.path.join(model_dir, "onnx_model.onnx")
        vec_path  = os.path.join(model_dir, "tfidf_vectorizer.pkl")

        for path in (onnx_path, vec_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required artefact missing: '{path}'. "
                    "Re-run export_onnx.py and repackage the .mar archive."
                )

        self.onnx_session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.vectorizer   = joblib.load(vec_path)
        self.initialized  = True

        logger.info("Handler initialised: ONNX session and vectorizer loaded.")

    # ------------------------------------------------------------------
    # Phase 2 of TorchServe lifecycle
    # ------------------------------------------------------------------

    def preprocess(self, data: List[Any]) -> np.ndarray:
        """
        Convert raw HTTP request bodies into a float32 TF-IDF matrix.

        What it does:
            1. Extracts text strings from the TorchServe data list.
               Accepts both {"text": "..."} JSON bodies and plain strings.
            2. Applies the same text cleaning as TextCleaner (reimplemented
               inline to avoid importing sklearn inside TorchServe).
            3. Runs the fitted TF-IDF vectorizer.
            4. Converts the sparse matrix to a dense float32 ndarray.

        Parameters:
            data (List[Any]): List of request payloads from TorchServe.

        Returns:
            np.ndarray: Shape (N, vocab_size), dtype float32.

        Connects to:
            src/text_preprocessing.py — mirrors TextCleaner.transform() logic.
            src/feature_engineering.py — uses the same TfidfVectorizer settings.
        """
        texts: List[str] = []

        for item in data:
            # TorchServe passes bytes or a dict; handle both
            if isinstance(item, (bytes, bytearray)):
                item = item.decode("utf-8")
            if isinstance(item, str):
                try:
                    parsed = json.loads(item)
                    text = parsed.get("text", item)
                except json.JSONDecodeError:
                    text = item
            elif isinstance(item, dict):
                text = item.get("body", item.get("text", ""))
                if isinstance(text, (bytes, bytearray)):
                    text = text.decode("utf-8")
                    try:
                        parsed = json.loads(text)
                        text = parsed.get("text", text)
                    except json.JSONDecodeError:
                        pass
            else:
                text = str(item)

            texts.append(self._clean_text(text))

        tfidf_matrix = self.vectorizer.transform(texts)
        return tfidf_matrix.toarray().astype(np.float32)

    # ------------------------------------------------------------------
    # Phase 3 of TorchServe lifecycle
    # ------------------------------------------------------------------

    def inference(self, features: np.ndarray) -> np.ndarray:
        """
        Run the ONNX model and return softmax probabilities.

        Parameters:
            features (np.ndarray): Float32 TF-IDF matrix, shape (N, vocab_size).

        Returns:
            np.ndarray: Softmax probabilities, shape (N, 2).
                        Column 0 = P(negative), column 1 = P(positive).

        Connects to:
            export_onnx.py — ONNX model was exported with input name
                             'tfidf_features' and output name 'logits'.
        """
        input_name = self.onnx_session.get_inputs()[0].name
        logits = self.onnx_session.run(None, {input_name: features})[0]
        # softmax for probabilities
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probs

    # ------------------------------------------------------------------
    # Phase 4 of TorchServe lifecycle
    # ------------------------------------------------------------------

    def postprocess(self, probs: np.ndarray) -> List[Dict[str, Any]]:
        """
        Convert probability arrays into structured JSON-serialisable dicts.

        Parameters:
            probs (np.ndarray): Shape (N, 2) softmax probabilities.

        Returns:
            List[dict]: One dict per sample:
                {
                    "prediction":   "positive" | "negative",
                    "confidence":   float,
                    "class_scores": {"negative": float, "positive": float}
                }

        Connects to:
            TorchServe framework — return value is JSON-serialised by TorchServe
            and sent back to the HTTP caller.
        """
        results = []
        for row in probs:
            class_idx   = int(np.argmax(row))
            results.append({
                "prediction":   LABEL_MAP[class_idx],
                "confidence":   float(row[class_idx]),
                "class_scores": {
                    "negative": float(row[0]),
                    "positive": float(row[1]),
                },
            })
        return results

    # ------------------------------------------------------------------
    # TorchServe unified entry point
    # ------------------------------------------------------------------

    def handle(self, data: List[Any], context) -> List[Dict[str, Any]]:
        """
        Single entry point called by TorchServe for every inference request.

        Parameters:
            data    (List[Any]): Raw request payloads.
            context:             TorchServe context (used by initialize).

        Returns:
            List[dict]: Postprocessed prediction dicts, one per input item.
        """
        if not self.initialized:
            self.initialize(context)

        features = self.preprocess(data)
        probs    = self.inference(features)
        return   self.postprocess(probs)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Replicate TextCleaner.transform() without importing sklearn.

        Mirrors src/text_preprocessing.py exactly:
            1. Lowercase
            2. Remove punctuation (replace non-word, non-space with space)
            3. Collapse whitespace

        Parameters:
            text (str): Raw review string.

        Returns:
            str: Cleaned text.
        """
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


# TorchServe expects a module-level callable named _service or the class
# itself to be instantiated.  Expose a module-level instance:
_service = SentimentHandler()


def handle(data, context):
    """
    Module-level handle function required by TorchServe.

    Delegates to the SentimentHandler singleton.
    """
    return _service.handle(data, context)
