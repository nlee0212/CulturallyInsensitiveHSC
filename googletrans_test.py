# Imports the Google Cloud Translation library
from google.cloud import translate

# Initialize Translation client
def translate_text(text="YOUR_TEXT_TO_TRANSLATE", project_id="optimum-task-376506"):
    """Translating Text."""

    client = translate.TranslationServiceClient()

    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    # Translate text from English to French
    # Detail on supported types can be found here:
    # https://cloud.google.com/translate/docs/supported-formats
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",  # mime types: text/plain, text/html
            "source_language_code": "ar",
            "target_language_code": "en-US",
        }
    )

    # Display the translation for each input text provided
    for translation in response.translations:
        print("Translated text: {}".format(translation.translated_text))

text = 'افطرت عليك بعقاء واثنين من فروخها الجن 🔪😂'
translate_text(text)