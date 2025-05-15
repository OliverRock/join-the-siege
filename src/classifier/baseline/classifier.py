from werkzeug.datastructures import FileStorage


class DocumentClassifier:
    def classify_file(self, file: FileStorage):
        filename = file.filename.lower()
        # file_bytes = file.read()

        if "drivers_license" in filename:
            return "drivers_licence"

        if "bank_statement" in filename:
            return "bank_statement"

        if "invoice" in filename:
            return "invoice"

        return "unknown file"
