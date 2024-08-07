from api.endpoints.search import retrieve_documents
from api.endpoints.generate import get_openai_answer, build_prompt
import argparse


def qa_bot(question: str, course: str = None) -> str:
    if course:
        filter = {"term": {"course": course}}
    else:
        filter = None

    response = retrieve_documents(query=question, max_results=3, filter=filter)
    prompt = build_prompt(question, response)
    answer = get_openai_answer(prompt)

    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--question", type=str, help="The question to ask")
    parser.add_argument("-f", "--filter", type=str, help="The course filter to apply")

    args = parser.parse_args()

    sample_question = args.question
    filter = {"course": args.filter} if args.filter else None

    response, scores = retrieve_documents(
        query=sample_question, max_results=3, filter=filter
    )

    print(f"scores: {scores}\n")
    for doc in response:
        print(f"course: {doc['course']}")
        print(f"section: {doc['section']}")
        print(f"question: {doc['question']}")
        print(f"answer: {doc['text'][:60]}...\n")


if __name__ == "__main__":
    main()
