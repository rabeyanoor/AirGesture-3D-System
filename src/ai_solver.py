import os

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class AISolver:
    """
    Gemini Vision AI integration for math/question solving.
    Sends typed text to Gemini and returns the answer.
    """

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY", "")
        self.enabled = False
        self.last_question = ""
        self.last_answer = ""

        if HAS_GENAI and api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                self.enabled = True
                print("[AI] Gemini AI solver initialized.")
            except Exception as e:
                print(f"[AI] Gemini init failed: {e}")
        else:
            print("[AI] Gemini AI solver inactive (missing package or GEMINI_API_KEY).")

    def solve(self, question: str) -> str:
        """Send question to Gemini and return concise answer."""
        if not self.enabled or not question.strip():
            return ""

        if question == self.last_question and self.last_answer:
            return self.last_answer

        try:
            prompt = (
                "You are a concise AR assistant. Answer briefly in 1-2 lines.\n"
                f"Question or equation: {question}"
            )
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            self.last_question = question
            self.last_answer = answer
            return answer
        except Exception as e:
            return f"[AI Error: {e}]"
