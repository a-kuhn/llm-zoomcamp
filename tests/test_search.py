import unittest
from unittest.mock import patch
from api.endpoints.search import retrieve_documents

class TestRetrieveDocuments(unittest.TestCase):

    @patch('api.endpoints.search.Elasticsearch')
    def test_retrieve_documents_with_query(self, MockElasticsearch):
        mock_es = MockElasticsearch.return_value
        mock_response = {
            "hits": {
                "hits": [
                    {"_source": {"question": "What is AI?", "text": "AI stands for Artificial Intelligence."}, "_score": 1.0},
                    {"_source": {"question": "How does AI work?", "text": "AI works by using algorithms and data."}, "_score": 0.9},
                ]
            }
        }
        mock_es.search.return_value = mock_response

        query = "What is AI?"
        documents, scores = retrieve_documents(query=query, index_name="faq_elasticsearch_data", max_results=5, filter=None)

        mock_es.search.assert_called_once()
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0]["question"], "What is AI?")
        self.assertEqual(scores[0], 1.0)

    @patch('api.endpoints.search.Elasticsearch')
    def test_retrieve_documents_with_filter(self, MockElasticsearch):
        mock_es = MockElasticsearch.return_value
        mock_response = {
            "hits": {
                "hits": [
                    {"_source": {"question": "How to use Docker?", "text": "Docker is a containerization platform."}, "_score": 1.2}
                ]
            }
        }
        mock_es.search.return_value = mock_response

        filter = {"course": "docker-course"}
        documents, scores = retrieve_documents(query=None, index_name="faq_elasticsearch_data", max_results=5, filter=filter)

        mock_es.search.assert_called_once()
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0]["question"], "How to use Docker?")
        self.assertEqual(scores[0], 1.2)

    @patch('api.endpoints.search.Elasticsearch')
    def test_retrieve_documents_no_query_no_filter(self, MockElasticsearch):
        mock_es = MockElasticsearch.return_value
        mock_response = {
            "hits": {
                "hits": [
                    {"_source": {"question": "What is AI?", "text": "AI stands for Artificial Intelligence."}, "_score": 1.0},
                    {"_source": {"question": "How does AI work?", "text": "AI works by using algorithms and data."}, "_score": 0.9},
                ]
            }
        }
        mock_es.search.return_value = mock_response

        documents, scores = retrieve_documents(query=None, index_name="faq_elasticsearch_data", max_results=5, filter=None)

        mock_es.search.assert_called_once()
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0]["question"], "What is AI?")
        self.assertEqual(scores[0], 1.0)

if __name__ == '__main__':
    unittest.main()
