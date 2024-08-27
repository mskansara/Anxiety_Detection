import unittest
from unittest.mock import patch


class TestTextGeneration(unittest.TestCase):

    @patch("lllama_test.text_generator")
    def test_get_response_non_empty(self, mock_text_generator):
        mock_text_generator.return_value = [
            {"generated_text": "This is a test summary."}
        ]
        prompt = "Test prompt"
        result = get_response(prompt)
        self.assertTrue(len(result) > 0, "The response should not be empty.")

    @patch("lllama_test.text_generator")
    def test_get_response_length(self, mock_text_generator):
        mock_text_generator.return_value = [
            {"generated_text": "This is a test summary."}
        ]
        prompt = "Test prompt"
        result = get_response(prompt)
        self.assertTrue(
            len(result) > 10, "The response should be longer than 10 characters."
        )

    @patch("lllama_test.text_generator")
    def test_get_response_no_error(self, mock_text_generator):
        mock_text_generator.return_value = [
            {"generated_text": "This is a test summary."}
        ]
        prompt = "Test prompt"
        try:
            result = get_response(prompt)
        except Exception as e:
            self.fail(f"get_response raised an exception {e}")


if __name__ == "__main__":
    unittest.main()
