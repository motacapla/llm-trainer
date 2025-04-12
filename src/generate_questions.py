from langchain_ollama import OllamaLLM
from typing import List
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()


class QuestionGenerator:
    def __init__(self):
        self.model = OllamaLLM(model="qwen2.5:7b")

    def generate_questions(
        self, requirements: str, resume: str, num_questions: int = 5
    ) -> List[str]:
        prompt = f"""
        以下の要件と候補者の情報から、面接で使用できる質問を{num_questions}個生成してください。
                
        要件:
        {requirements}

        候補者の情報:
        {resume}

        生成する質問は以下の条件を満たす必要があります：
        1. 要件と候補者の経験を関連付けた具体的な質問
        2. 技術的な深さを確認できる質問
        3. 実務経験に基づいた質問
        4. オープンエンドな質問（Yes/Noで答えられない質問）

        質問を箇条書きで出力してください。
        """
        response = self.model.invoke(prompt)
        questions = [q.strip() for q in response.split("\n") if q.strip()]
        return questions[:num_questions]

    def save_questions_to_file(
        self, questions: List[str], requirements: str, resume: str
    ):
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/questions_{timestamp}.txt"

        with open(filename, "w", encoding="utf-8") as f:
            f.write("=== 要件 ===\n")
            f.write(requirements)
            f.write("\n\n=== 候補者の情報 ===\n")
            f.write(resume)
            f.write("\n\n=== 生成された質問 ===\n")
            for q in questions:
                f.write(f"{q}\n")

        return filename


def main():
    generator = QuestionGenerator()

    requirements = """
    - 5年以上のPython開発経験
    - 大規模なWebアプリケーションの開発経験
    - マイクロサービスアーキテクチャの実装経験
    - クラウドインフラ（AWS/GCP）の知識
    """

    resume = """
    7年のPython開発経験があります。主にDjangoを使用したWebアプリケーション開発に従事。
    最近はAWSを使用したマイクロサービスアーキテクチャの設計と実装を担当。
    チームリーダーとして5人のチームを率いた経験あり。
    """

    questions = generator.generate_questions(requirements, resume, num_questions=30)
    output_file = generator.save_questions_to_file(questions, requirements, resume)
    print(f"質問が{output_file}に保存されました。")


if __name__ == "__main__":
    main()
