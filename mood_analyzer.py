# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS

# Words that signal a genuine "on the one hand... on the other hand" mix.
CONTRAST_WORDS = {"but", "though", "although", "yet", "however"}

# Words that flip the polarity of a nearby sentiment word.
NEGATORS = {
    "not", "no", "never", "dont", "don't", "isnt", "isn't", "cant", "can't",
    "wont", "won't", "didnt", "didn't", "wasnt", "wasn't", "cannot",
}


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        cleaned = text.strip().lower()
        # Keep emoticons (":)", ":-(") and apostrophes ("don't") intact, grab
        # words, and treat any other symbol (emoji) as its own token.
        raw = re.findall(r"[:;]-?[()dp]|\w+(?:'\w+)?|[^\w\s]", cleaned)
        # Drop plain ASCII punctuation, keep emoticons and emoji.
        tokens = [t for t in raw if len(t) > 1 or t.isalnum() or ord(t) > 127]

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        positive_hits, negative_hits = self._find_hits(text)
        return len(positive_hits) - len(negative_hits)

    def _find_hits(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Return (positive_hits, negative_hits) for the text.

        Enhancement: negation handling. A negator ("not", "never", "don't", ...)
        flips the polarity of a sentiment word within the next two tokens,
        so "not happy" counts as negative and "not bad" counts as positive.
        """
        tokens = self.preprocess(text)
        positive_hits: List[str] = []
        negative_hits: List[str] = []
        negate_for = 0  # how many more tokens a negator still applies to

        for i, token in enumerate(tokens):
            if token in NEGATORS:
                # "no cap" is slang for "for real", not a negation.
                if token == "no" and tokens[i + 1:i + 2] == ["cap"]:
                    continue
                negate_for = 2
                continue

            is_pos = token in self.positive_words
            is_neg = token in self.negative_words
            if negate_for > 0:
                if is_pos or is_neg:
                    is_pos, is_neg = is_neg, is_pos
                    token = f"not {token}"
                    negate_for = 0
                else:
                    negate_for -= 1

            if is_pos:
                positive_hits.append(token)
            elif is_neg:
                negative_hits.append(token)

        return positive_hits, negative_hits

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"
        with an override when both positive and negative words appear
        (see the comments below).

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        positive_hits, negative_hits = self._find_hits(text)

        if positive_hits and negative_hits:
            # Both polarities present. A contrast word ("but") means the writer
            # really feels both; without one, praise next to a bad event is
            # usually sarcasm ("I love getting stuck in traffic"), so lean negative.
            if any(t in CONTRAST_WORDS for t in self.preprocess(text)):
                return "mixed"
            return "negative"

        score = len(positive_hits) - len(negative_hits)
        if score > 0:
            return "positive"
        if score < 0:
            return "negative"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        positive_hits, negative_hits = self._find_hits(text)
        score = len(positive_hits) - len(negative_hits)

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )
