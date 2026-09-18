"""Stage 5: Devanagari language identification.

Hindi and Nepali share a script and a large Sanskrit-derived vocabulary, and the
``ne`` slices of CC-100 / mC4 / CulturaX are known to contain Hindi (and the
reverse).  The project forbids sharing documents across the two corpora, so a
classifier is load-bearing infrastructure, not an optional filter.

Design decisions:

* **Six classes, not two.**  hi, ne, mr, bh, mai, sa.  A binary hi/ne classifier
  has nowhere to put Bhojpuri or Maithili and will confidently mislabel them as
  one of the two targets.  With six classes those documents are rejected.
* **Trained by us, from Wikipedia.**  Character 3-5-gram TF-IDF + logistic
  regression.  No pretrained model is used anywhere, consistent with the project
  constraints, and the held-out confusion matrix is a reportable result.
* **Abstain, do not guess.**  Documents below ``min_conf`` go to ``quarantine/``
  rather than being deleted, so per-source contamination rates can be reported.

Discriminative function words the model learns (report table):
  Nepali  छ / छन् / थियो / र / भन्दा / ले / लाई / गरेको
  Hindi   है / हैं / था / और / से / ने / को / किया
"""

from __future__ import annotations

import pickle
from pathlib import Path

DEVANAGARI_CLASSES = ["hi", "ne", "mr", "bh", "mai", "sa"]
WIKI_CONFIGS = {
    "hi": "20231101.hi",
    "ne": "20231101.ne",
    "mr": "20231101.mr",
    "bh": "20231101.bh",
    "mai": "20231101.mai",
    "sa": "20231101.sa",
}
MAX_CHARS_FOR_LID = 5000


def collect_wikipedia_samples(
    lang: str, n_samples: int = 20_000, min_chars: int = 200, max_chars: int = 2000
) -> list[str]:
    """Stream Wikipedia paragraphs for one language as LID training data.

    Paragraph-level samples (not whole articles) match the granularity at which
    contamination actually occurs and give a larger, more varied training set.
    """
    from datasets import load_dataset

    ds = load_dataset(
        "wikimedia/wikipedia", WIKI_CONFIGS[lang], split="train", streaming=True
    )
    out: list[str] = []
    for row in ds:
        for para in row["text"].split("\n\n"):
            para = para.strip()
            if min_chars <= len(para) <= max_chars:
                out.append(para)
                if len(out) >= n_samples:
                    return out
    return out


def build_training_set(
    langs=DEVANAGARI_CLASSES, n_per_class: int = 20_000
) -> tuple[list[str], list[str]]:
    """Assemble a balanced (texts, labels) set across the Devanagari classes.

    Classes with small Wikipedias (bh, mai) contribute fewer samples; the
    classifier is fitted with balanced class weights to compensate.
    """
    X, y = [], []
    for lang in langs:
        samples = collect_wikipedia_samples(lang, n_per_class)
        X.extend(samples)
        y.extend([lang] * len(samples))
        print(f"[langid] {lang}: {len(samples)} paragraphs")
    return X, y


def train(X: list[str], y: list[str], out_path: str | Path, seed: int = 1337) -> dict:
    """Fit the char n-gram classifier and write it plus its evaluation report.

    Returns:
        Dict with the classification report and confusion matrix, for the report.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.1, random_state=seed, stratify=y
    )
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=200_000,
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000, C=4.0, class_weight="balanced", n_jobs=-1
                ),
            ),
        ]
    )
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    labels = sorted(set(y))
    result = {
        "labels": labels,
        "classification_report": classification_report(
            yte, pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(yte, pred, labels=labels).tolist(),
        "n_train": len(Xtr),
        "n_test": len(Xte),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(pipe, fh)
    return result


class LanguageIdentifier:
    """Applies a trained Devanagari classifier with an abstain threshold."""

    def __init__(self, model_path: str | Path, target: str, min_conf: float = 0.90):
        with open(model_path, "rb") as fh:
            self.pipe = pickle.load(fh)
        self.target = target
        self.min_conf = min_conf
        self.classes = list(self.pipe.classes_)

    def predict(self, text: str) -> tuple[str, float]:
        """Return (predicted language, confidence) for one document.

        Convenience wrapper over :meth:`predict_many`.  Prefer the batched form
        in pipeline code: a one-document ``predict_proba`` call spends nearly all
        of its time on scipy sparse-matrix setup rather than on the document, so
        per-document calls run one to two orders of magnitude slower than
        batching the same texts.
        """
        return self.predict_many([text])[0]

    def predict_many(self, texts: list[str]) -> list[tuple[str, float]]:
        """Vectorised (predicted language, confidence) for a batch of documents."""
        if not texts:
            return []
        probs = self.pipe.predict_proba([t[:MAX_CHARS_FOR_LID] for t in texts])
        best = probs.argmax(axis=1)
        return [(self.classes[int(j)], float(probs[i, j])) for i, j in enumerate(best)]

    def accept(self, doc) -> bool:
        """True when the document is confidently in the target language.

        Records the prediction on the document so quarantined items keep a
        machine-readable reason.
        """
        return self.accept_many([doc])[0]

    def accept_many(self, docs: list) -> list[bool]:
        """Batched :meth:`accept`.  Stamps ``lid_pred``/``lid_conf`` on each doc."""
        out = []
        for doc, (lang, conf) in zip(docs, self.predict_many([d.text for d in docs])):
            doc.flags["lid_pred"] = lang
            doc.flags["lid_conf"] = round(conf, 4)
            out.append(lang == self.target and conf >= self.min_conf)
        return out
