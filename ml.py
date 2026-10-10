"""AI/ML module for the Smart Library system.

Everything here is plain Python (math + collections). No numpy, pandas or
scikit-learn is needed, so it runs on locked-down machines where compiled
packages cannot be loaded. The maths is the same as the synopsis describes:
TF-IDF vectors compared with cosine similarity.
"""
import math
import re
from collections import Counter

STOPWORDS = set("""a an the and or of to in for with on at by from is are was be it its as this that these those
i me my we you your want need looking book books about how what which some any can please give show find
good best new""".split())

# Small synonym map so a plain-English query reaches books that use different words.
SYNONYMS = {
    "easy": ["introduction", "fundamentals", "beginner", "guide"],
    "simple": ["introduction", "fundamentals", "guide"],
    "beginner": ["introduction", "fundamentals", "guide"],
    "beginners": ["introduction", "fundamentals", "guide"],
    "basic": ["introduction", "fundamentals"],
    "basics": ["introduction", "fundamentals"],
    "starter": ["introduction", "fundamentals"],
    "learn": ["introduction", "fundamentals", "guide"],
    "study": ["introduction", "fundamentals"],
    "understand": ["introduction", "fundamentals"],
    "ai": ["artificial", "intelligence", "machine", "learning"],
    "ml": ["machine", "learning"],
    "coding": ["programming", "code", "software"],
    "code": ["programming", "software"],
    "programming": ["code", "software", "language"],
    "database": ["sql", "data", "query"],
    "databases": ["sql", "data", "query"],
    "sql": ["database", "query"],
    "web": ["javascript", "html", "browser"],
    "website": ["web", "javascript", "html"],
    "network": ["networks", "internet", "protocol"],
    "networking": ["network", "internet", "protocol"],
    "internet": ["network", "protocol"],
    "os": ["operating", "system", "kernel"],
    "algorithm": ["algorithms", "data", "structures"],
    "algorithms": ["data", "structures", "complexity"],
    "dsa": ["algorithms", "data", "structures"],
    "deep": ["neural", "learning"],
    "neural": ["deep", "learning"],
    "math": ["mathematics", "discrete", "logic"],
    "maths": ["mathematics", "discrete", "logic"],
    "mathematics": ["math", "discrete", "logic"],
    "data": ["database", "analysis"],
    "analysis": ["data", "statistics"],
    "search": ["retrieval"],
}


def stem(word):
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)] + replacement
    return word


def tokens(text):
    words = re.findall(r"[a-z0-9+#]+", (text or "").lower())
    return [stem(w) for w in words if w not in STOPWORDS]


def cosine(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(weight * b.get(term, 0.0) for term, weight in a.items())


def normalise(vec):
    length = math.sqrt(sum(v * v for v in vec.values()))
    return {t: v / length for t, v in vec.items()} if length else {}


class Index:
    """TF-IDF index over the book catalogue. Each book is a dict with id, title, author, category, description."""

    def __init__(self, books):
        self.books = {b["id"]: b for b in books}
        docs = {}
        for b in books:
            text = " ".join([b["title"], b["title"], b["category"], b["category"], b["author"], b.get("description") or ""])
            docs[b["id"]] = Counter(tokens(text))
        n = len(docs)
        df = Counter()
        for counts in docs.values():
            df.update(counts.keys())
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self.vectors = {
            book_id: normalise({t: (1 + math.log(c)) * self.idf[t] for t, c in counts.items()})
            for book_id, counts in docs.items()
        }

    def _query_vector(self, query):
        weights = Counter()
        for raw in re.findall(r"[a-z0-9+#]+", query.lower()):
            if raw in STOPWORDS:
                continue
            weights[stem(raw)] += 1.0
            for extra in SYNONYMS.get(raw, []):
                weights[stem(extra)] += 0.4
        return normalise({t: w * self.idf[t] for t, w in weights.items() if t in self.idf})

    def search(self, query, limit=20):
        """Natural-language search: rank books by cosine similarity to the query."""
        qvec = self._query_vector(query)
        phrase = query.strip().lower()
        results = []
        for book_id, vec in self.vectors.items():
            score = cosine(qvec, vec)
            if phrase and phrase in self.books[book_id]["title"].lower():
                score += 0.3
            if score > 0.04:
                results.append((book_id, score))
        results.sort(key=lambda r: r[1], reverse=True)
        return results[:limit]

    def similar(self, book_id, limit=4):
        base = self.vectors.get(book_id)
        if not base:
            return []
        scored = [(other, cosine(base, vec)) for other, vec in self.vectors.items() if other != book_id]
        scored = [s for s in scored if s[1] > 0.05]
        scored.sort(key=lambda r: r[1], reverse=True)
        return scored[:limit]

    def recommend(self, profile, exclude, limit=8):
        """Content-based recommendations.

        profile: {book_id: weight} built from the student's borrowing (high weight) and viewing (low weight).
        Returns [(book_id, score, source_book_id)] where source is the profile book that matched best.
        """
        known = {b: w for b, w in profile.items() if b in self.vectors}
        if not known:
            return []
        user_vec = {}
        for book_id, weight in known.items():
            for term, value in self.vectors[book_id].items():
                user_vec[term] = user_vec.get(term, 0.0) + weight * value
        user_vec = normalise(user_vec)
        results = []
        for book_id, vec in self.vectors.items():
            if book_id in exclude:
                continue
            score = cosine(user_vec, vec)
            if score <= 0.03:
                continue
            source = max(known, key=lambda k: cosine(self.vectors[k], vec))
            results.append((book_id, score, source))
        results.sort(key=lambda r: r[1], reverse=True)
        return results[:limit]


def predict_demand(books, loans, now, loan_days=14, weeks=8):
    """Forecast next-30-day demand per book from the last `weeks` weeks of issues.

    A weighted moving average gives recent weeks more influence. One copy can serve about
    30 / loan_days borrowers per month, which tells the librarian how many copies are needed.
    books: [{id, title, category, total}], loans: [(book_id, issued_datetime)]
    """
    buckets = {b["id"]: [0] * weeks for b in books}
    for book_id, issued in loans:
        age = (now - issued).days
        if book_id in buckets and 0 <= age < 7 * weeks:
            buckets[book_id][weeks - 1 - age // 7] += 1
    weights = list(range(1, weeks + 1))
    rows = []
    for b in books:
        counts = buckets[b["id"]]
        if not sum(counts):
            continue
        weekly = sum(c * w for c, w in zip(counts, weights)) / sum(weights)
        predicted = round(weekly * 30 / 7, 1)
        older, recent = sum(counts[: weeks // 2]), sum(counts[weeks // 2:])
        trend = "rising" if recent > older else "falling" if recent < older else "steady"
        needed = math.ceil(predicted / (30 / loan_days))
        rows.append({
            "book_id": b["id"], "title": b["title"], "category": b["category"],
            "issued_last_8_weeks": sum(counts), "predicted_30_days": predicted, "trend": trend,
            "copies": b["total"], "extra_copies": max(0, needed - b["total"]),
        })
    rows.sort(key=lambda r: r["predicted_30_days"], reverse=True)
    return rows[:15]
