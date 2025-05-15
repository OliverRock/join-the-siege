import pytest

# Import your DocumentTextExtractor class
from src.text_extraction.text_extraction import DocumentTextExtractor


class TestDocumentTextExtractor:
    """Pytest test suite for DocumentTextExtractor"""

    @pytest.fixture(scope="class")
    def extractor(self):
        """Fixture to create and return the extractor instance"""
        return DocumentTextExtractor()

    # Simple test parameters for text and HTML files
    @pytest.mark.parametrize(
        "filename,expected_content",
        [
            ("files/bank_statement_1.pdf", "Bank 1 - Confidential Statement | Page 2"),
            ("files/bank_statement_2.pdf", "Account Number: XXXX-XXXX-XXXX-6782"),
            ("files/bank_statement_3.pdf", "Debit Card Purchase"),
            ("files/drivers_license_1.jpg", "ANYTOWN, ANY STATE"),
            ("files/drivers_licence_2.jpg", "DRIVING LICENCE"),
            ("files/drivers_license_3.jpg", "HONOLULU"),
            ("files/invoice_1.pdf", "Customer Company Name "),
            ("files/invoice_2.pdf", "MORGAN MAXWELL"),
            ("files/invoice_3.pdf", "CONDITIONS/INSTRUCTIONS"),
        ],
    )
    def test_text_extraction_basic(self, extractor, filename, expected_content):
        """Test basic text extraction for text and HTML files"""
        # Read file data
        with open(filename, "rb") as f:
            file_data = f.read()

        # Extract text
        extracted_text = extractor.extract_text(file_data, filename)

        # Compare with expected output (strip to remove any extra whitespace)
        assert expected_content.strip() in extracted_text.strip()

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("file.pdf", True),
            ("file.png", True),
            ("file.jpg", True),
            ("file.txt", True),
            ("file", False),
        ],
    )
    def test_allowed_file(self, extractor, filename, expected):
        assert extractor.allowed_file(filename) == expected
