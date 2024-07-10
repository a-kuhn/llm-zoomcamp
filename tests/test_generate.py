import pytest
from unittest.mock import patch, MagicMock
from api.endpoints.generate import (
    get_openai_answer,
    build_prompt,
    build_context,
)

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


@patch("api.endpoints.generate.OpenAI")
def test_get_openai_answer(mock_openai_class):
    mock_openai_instance = MagicMock()
    mock_openai_class.return_value = mock_openai_instance
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Mocked answer from OpenAI"
    mock_openai_instance.chat.completions.create.return_value = mock_response

    prompt = "What is the capital of France?"
    result = get_openai_answer(prompt)

    mock_openai_instance.chat.completions.create.assert_called_once_with(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    )
    assert result == "Mocked answer from OpenAI"


def test_build_prompt():
    question = "How do I execute a command in a running docker container?"
    result = build_prompt(question, mock_context_docs)
    assert result == mock_prompt


def test_build_context_with_valid_docs():
    expected_output = (
        "Section: 1.1\nQuestion: What is Docker?\nAnswer: Docker is a tool...\n\n"
        "Section: 1.2\nQuestion: How to use Docker?\nAnswer: To use Docker..."
    )
    assert build_context(mock_context_docs) == expected_output


def test_build_context_with_empty_list():
    assert build_context([]) == ""


def test_build_context_with_non_list():
    assert build_context("Not a list") == "Error: context_docs should be a list."


def test_build_context_with_partial_valid_docs():
    invalid_docs = [
        {
            "section": "1.1",
            "question": "What is Docker?",
            "text": "Docker is a tool...",
        },
        "This is a string, not a dict",
    ]
    assert (
        build_context(invalid_docs)
        == "Error: Each item in context_docs should be a dictionary."
    )


if __name__ == "__main__":
    pytest.main()
