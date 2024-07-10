import pytest
from unittest.mock import patch

from api.app import qa_bot, main

mock_context_docs = [
    {
        "course": "machine-learning-zoomcamp",
        "section": "1.1",
        "question": "What is Docker?",
        "text": "Docker is a tool...",
    },
    {
        "course": "machine-learning-zoomcamp",
        "section": "1.2",
        "question": "How to use Docker?",
        "text": "To use Docker...",
    },
]

mock_prompt = """
You're a course teaching assistant. Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database. 
Only use the facts from the CONTEXT. If the CONTEXT doesn't contain the answer, return "NONE"

QUESTION: How do I execute a command in a running docker container?

CONTEXT:
Section: 1.1
Question: What is Docker?
Answer: Docker is a tool...

Section: 1.2
Question: How to use Docker?
Answer: To use Docker...
""".strip()

mock_answer = "You can use the `docker exec` command to execute a command in a running Docker container."


@patch("api.app.retrieve_documents")
@patch("api.app.build_prompt")
@patch("api.app.get_openai_answer")
def test_qa_bot_with_course(
    mock_get_openai_answer, mock_build_prompt, mock_retrieve_documents
):
    mock_retrieve_documents.return_value = mock_context_docs
    mock_build_prompt.return_value = mock_prompt
    mock_get_openai_answer.return_value = "Mocked answer from OpenAI"

    question = "How do I execute a command in a running docker container?"
    course = "machine-learning-zoomcamp"
    result = qa_bot(question, course)

    mock_retrieve_documents.assert_called_once_with(
        query=question, max_results=3, filter={"term": {"course": course}}
    )
    mock_build_prompt.assert_called_once_with(question, mock_context_docs)
    mock_get_openai_answer.assert_called_once_with(mock_prompt)

    assert result == "Mocked answer from OpenAI"


@patch("api.app.retrieve_documents")
@patch("api.app.build_prompt")
@patch("api.app.get_openai_answer")
def test_qa_bot_without_course(
    mock_get_openai_answer, mock_build_prompt, mock_retrieve_documents
):
    mock_retrieve_documents.return_value = mock_context_docs
    mock_build_prompt.return_value = mock_prompt
    mock_get_openai_answer.return_value = "Mocked answer from OpenAI"

    question = "How do I execute a command in a running docker container?"
    result = qa_bot(question)

    mock_retrieve_documents.assert_called_once_with(
        query=question, max_results=3, filter=None
    )
    mock_build_prompt.assert_called_once_with(question, mock_context_docs)
    mock_get_openai_answer.assert_called_once_with(mock_prompt)

    assert result == "Mocked answer from OpenAI"


@patch("api.app.retrieve_documents")
@patch("api.app.build_prompt")
@patch("api.app.get_openai_answer")
def test_main(mock_get_openai_answer, mock_build_prompt, mock_retrieve_documents):
    mock_retrieve_documents.return_value = (mock_context_docs, [0.9, 0.8])
    mock_build_prompt.return_value = mock_prompt
    mock_get_openai_answer.return_value = "Mocked answer from OpenAI"

    main()

    mock_retrieve_documents.assert_called_once_with(
        query="How do I execute a command in a running docker container?",
        max_results=3,
        filter={"course": "machine-learning-zoomcamp"},
    )


if __name__ == "__main__":
    pytest.main()
