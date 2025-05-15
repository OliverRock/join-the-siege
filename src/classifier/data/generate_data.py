import json
import os
import time
from pathlib import Path

from openai import OpenAI

# Set your OpenAI key
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

client = OpenAI(api_key=API_KEY)


default_query = (
    " The response should simulate an OCR extraction from the document. "
    "There may be minor errors or missing words. "
    "Make the response with high variance so that if you do it twice "
    "they will be very different. Response must be in English. "
    "Only include text from the document, do not add extra notes or explanations."
)

# Define your document classes and prompt templates
document_classes = {  # noqa
    "passport": "Generate the full text content of a fake passport. Include fields like name, nationality, date of birth, issue date, and expiry date. Make it look like a real document. Make it seem very real, do not make up country names.",  # noqa
    "bank_statement": "Generate a realistic fake bank statement for a customer. Include dates, transactions, balances, and account holder info.",  # noqa
    "utility_bill": "Generate the full text of a fake electricity utility bill. Include the customer's name, address, billing period, total amount, and usage in kWh.",  # noqa
    "insurance_policy": "Generate the body of a fake insurance policy document. Include policy number, coverage details, dates, conditions and other common things.",  # noqa
    "drivers_license": "Generate the full text content of a fake driver's license. Include fields like name, address, date of birth, issue date, expiry date, and license number.",  # noqa
    "invoice": "Generate a realistic fake invoice. Include invoice number, date, billing address, itemized list of services/products, total amount due, and payment terms.",  # noqa
    "unknown": "Generate some random text which you might find in a formal, business or official document.",  # noqa
}

# Parameters
num_samples_per_class = 100
output_dir = "data"
model = "gpt-4.1-mini"  # or "gpt-3.5-turbo"

Path(output_dir).mkdir(exist_ok=True)


def generate_sample(doc_type, prompt):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that writes realistic legal or financial documents.",  # noqa
                },
                {"role": "user", "content": prompt + default_query},
            ],
            temperature=1.2,
            max_tokens=600,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating {doc_type} sample: {e}")
        return None


def main():
    samples = []
    for doc_type, prompt in document_classes.items():
        print(f"\nGenerating samples for: {doc_type}")

        for i in range(num_samples_per_class):
            content = generate_sample(doc_type, prompt)
            if content:
                samples.append({"text": content, "label": doc_type})
                print(f"Generated {doc_type} sample {i+1}/{num_samples_per_class}")
            time.sleep(1.2)  # avoid rate limit

    with open(f"{output_dir}/samples.json", "w") as f:
        json.dump(samples, f, indent=2)


if __name__ == "__main__":
    main()
