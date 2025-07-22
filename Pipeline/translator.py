from deep_translator import GoogleTranslator

language_map = {
    'english': 'en',
    'danish': 'da',
    'german': 'de',
    'dutch': 'nl',
    'swedish': 'sv',
    'spanish': 'es',
    'french': 'fr',
    'italian': 'it',
    'portuguese': 'pt',
    'romanian': 'ro',
    'bulgarian': 'bg',
    'czech': 'cs',
    'croatian': 'hr',
    'polish': 'pl',
    'slovenian': 'sl',
    'estonian': 'et',
    'finnish': 'fi',
    'hungarian': 'hu',
    'lithuanian': 'lt',
    'latvian': 'lv',
    'greek': 'el',
    'irish': 'ga',
    'maltese': 'mt',
    'slovak': 'sk',
    'chinese': 'zh-CN',
    'zh': 'zh-CN',
    "nb": "no",  # Norwegian Bokmål → deep-translator uses 'no'
}


def translate(target_language, inst):
    if target_language in language_map:
        target_language = language_map[target_language]
    if target_language == 'en':
        return inst
    translator = GoogleTranslator(source='en', target=target_language)
    return translator.translate(inst)