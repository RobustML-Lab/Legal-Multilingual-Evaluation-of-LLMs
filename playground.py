from deep_translator import GoogleTranslator

away = GoogleTranslator(source="en", target="ar")
home = GoogleTranslator(source="ar", target="en")

word = "entailment"
print(home.translate(away.translate(word)))
