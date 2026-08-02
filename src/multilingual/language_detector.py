"""
language_detector.py
----------------------
Automatic language identification using `langdetect` (Python port of
Google's language-detection library). Fully offline - no API calls needed.
"""
from langdetect import detect, detect_langs, DetectorFactory, LangDetectException

DetectorFactory.seed = 42  # deterministic results

LANGUAGE_NAMES = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German",
    "it": "Italian", "pt": "Portuguese", "zh-cn": "Chinese", "ja": "Japanese",
    "ar": "Arabic", "ru": "Russian", "hi": "Hindi", "nl": "Dutch", "ko": "Korean",
}


class LanguageDetector:
    def detect(self, text: str) -> dict:
        if not text or not str(text).strip():
            return {"language_code": "unknown", "language_name": "Unknown", "confidence": 0.0}
        try:
            candidates = detect_langs(text)
            best = candidates[0]
            code = best.lang
            return {
                "language_code": code,
                "language_name": LANGUAGE_NAMES.get(code, code),
                "confidence": round(best.prob, 3),
            }
        except LangDetectException:
            return {"language_code": "unknown", "language_name": "Unknown", "confidence": 0.0}

    def detect_dataframe(self, df, text_col="content"):
        df = df.copy()
        detections = df[text_col].apply(self.detect)
        df["language_code"] = detections.apply(lambda d: d["language_code"])
        df["language_name"] = detections.apply(lambda d: d["language_name"])
        df["language_confidence"] = detections.apply(lambda d: d["confidence"])
        return df

    def language_distribution(self, df, lang_col="language_name"):
        return df[lang_col].value_counts()
