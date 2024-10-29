from deep_translator import GoogleTranslator


def translate(target_language, inst):
    translator = GoogleTranslator(source='en', target=target_language)
    return translator.translate(inst), translator.translate("Text")