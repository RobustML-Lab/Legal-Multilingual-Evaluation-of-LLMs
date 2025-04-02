import json
import time
import pandas as pd
import re
import requests

# --------------------------
# CONFIGURATION
# --------------------------

languages = ["fr", "ar" ] #"es", "ru", "zh" "en",
model_name = "llama3.1:8b"
file_path = "sampled_sdgs_attack_character_level_02.jsonl"

OLLAMA_API = "http://localhost:11434/api/generate"

# --------------------------
# LANGUAGE-SPECIFIC SDG LABELS
# --------------------------

subjects_en = [
"1 - NO POVERTY",
"2 - ZERO HUNGER",
"3 - GOOD HEALTH AND WELL-BEING",
"4 - QUALITY EDUCATION",
"5 - GENDER EQUALITY",
"6 - CLEAN WATER AND SANITATION",
"7 - AFFORDABLE AND CLEAN ENERGY",
"8 - DECENT WORK AND ECONOMIC GROWTH",
"9 - INDUSTRY, INNOVATION AND INFRASTRUCTURE",
"10 - REDUCED INEQUALITIES",
"11 - SUSTAINABLE CITIES AND COMMUNITIES",
"12 - RESPONSIBLE CONSUMPTION AND PRODUCTION",
"13 - CLIMATE ACTION",
"14 - LIFE BELOW WATER",
"15 - LIFE ON LAND",
"16 - PEACE, JUSTICE AND STRONG INSTITUTIONS",
"17 - PARTNERSHIPS FOR THE GOALS"

]

subjects_ar = [
    "1 - القضاء على الفقر",
    "2 - القضاء على الجوع",
    "3 - الصحة الجيدة والرفاهية",
    "4 - التعليم الجيد",
    "5 - المساواة بين الجنسين",
    "6 - نظافة المياه والصرف الصحي",
    "7 - طاقة نظيفة وبتكلفة ميسورة",
    "8 - عمل لائق ونمو اقتصادي",
    "9 - صناعة، ابتكار وبنية التحتية",
    "10 - الحد من انعدام المساواة",
    "11 - مدن ومجتمعات مستدامة",
    "12 - استهلاك وإنتاج مسؤول",
    "13 - العمل المناخي",
    "14 - الحياة تحت الماء",
    "15 - الحياة البرية",
    "16 - سلام، عدالة ومؤسسات قوية",
    "17 - شراكات من أجل الأهداف"
]

subjects_ru = [
    "1 - ЛИКВИДАЦИЯ НИЩЕТЫ",
    "2 - ЛИКВИДАЦИЯ ГОЛОДА",
    "3 - ХОРОШЕЕ ЗДОРОВЬЕ И БЛАГОПОЛУЧИЕ",
    "4 - КАЧЕСТВЕННОЕ ОБРАЗОВАНИЕ",
    "5 - ГЕНДЕРНОЕ РАВЕНСТВО",
    "6 - ЧИСТАЯ ВОДА И САНИТАРИЯ",
    "7 - НЕДОРОГОСТОЯЩАЯ И ЧИСТАЯ ЭНЕРГИЯ",
    "8 - ДОСТОЙНАЯ РАБОТА И ЭКОНОМИЧЕСКИЙ РОСТ",
    "9 - ИНДУСТРИАЛИЗАЦИЯ, ИННОВАЦИЯ И ИНФРАСТРУКТУРА",
    "10 - УМЕНЬШЕНИЕ НЕРАВЕНСТВА",
    "11 - УСТОЙЧИВЫЕ ГОРОДА И НАСЕЛЕННЫЕ ПУНКТЫ",
    "12 - ОТВЕТСТВЕННОЕ ПОТРЕБЛЕНИЕ И ПРОИЗВОДСТВО",
    "13 - БОРЬБА С ИЗМЕНЕНИЕМ КЛИМАТА",
    "14 - СОХРАНЕНИЕ МОРСКИХ ЭКОСИСТЕМ",
    "15 - СОХРАНЕНИЕ ЭКОСИСТЕМ СУШИ",
    "16 - МИР, ПРАВОСУДИЕ И ЭФФЕКТИВНЫЕ ИНСТИТУТЫ",
    "17 - ПАРТНЕРСТВО В ИНТЕРЕСАХ УСТОЙЧИВОГО РАЗВИТИЯ"
]

subjects_es = [
    "1 - Pobreza",
    "2 - Cero Hambre",
    "3 - Buena Salud y Bienestar",
    "4 - Educación de Calidad",
    "5 - Igualdad de Género",
    "6 - Agua potable y Saneamiento",
    "7 - Energía Limpia y Accesible",
    "8 - Trabajo Decente y Crecimiento Económico",
    "9 - Industria, Innovación e Infraestructura",
    "10 - Desigualdades Reducidas",
    "11 - Ciudades y Comunidades Sostenibles",
    "12 - Producción y Consumo Responsables",
    "13 - Cambio Climático",
    "14 - Océanos",
    "15 - Bosques, desertificación y diversidad biológica",
    "16 - Paz, Justicia e Instituciones",
    "17 - Alianzas para los Objetivos"
]

subjects_fr = [
    "1 - PAS DE PAUVRETE",
    "2 - FAIM \"ZERO\"",
    "3 - BONNE SANTE ET BIEN-ETRE",
    "4 - EDUCATION DE QUALITE",
    "5 - EGALITE ENTRE LES SEXES",
    "6 - EAU PROPRE ET ASSAINISSEMENT",
    "7 - ENERGIE PROPRE ET D'UN COUT ABORDABLE",
    "8 - TRAVAIL DECENT ET CROISSANCE ECONOMIQUE",
    "9 - INDUSTRIE, INNOVATION ET INFRASTRUCTURE",
    "10 - INEGALITES REDUITES",
    "11 - VILLES ET COMMUNAUTES DURABLES",
    "12 - CONSOMMATION ET PRODUCTION RESPONSABLES",
    "13 - LUTTE CONTRE LES CHANGEMENTS CLIMATIQUES",
    "14 - VIE AQUATIQUE",
    "15 - VIE TERRESTRE",
    "16 - PAIX, JUSTICE ET INSTITUTIONS EFFICACES",
    "17 - PARTENARIATS POUR LA REALISATION DES OBJECTIFS"
]

subjects_ch = [
    "1. 无贫穷",
    "2. 零饥饿",
    "3. 良好健康与福祉",
    "4. 优质教育",
    "5. 性别平等",
    "6. 清洁饮水和卫生设施",
    "7. 可负担的清洁能源",
    "8. 体面工作和经济增长",
    "9. 产业、创新和基础设施",
    "10. 减少不平等",
    "11. 可持续城市和社区",
    "12. 负责任消费和生产",
    "13. 气候行动",
    "14. 水下生物",
    "15. 陆地生物",
    "16. 和平、正义与强大机构",
    "17. 促进目标实现的伙伴关系"
]

subjects_dict = {
    "en": subjects_en,
    "ar": subjects_ar,
    "es": subjects_es,
    "fr": subjects_fr,
    "ru": subjects_ru,
    "zh": subjects_ch
}

for language in languages:
    csv_filename = f"llama_character_{language}.csv"
    subjects = subjects_dict[language]
    subjects_str = ", ".join(subjects)

    # --------------------------
    # OLLAMA CALL
    # --------------------------

    def query_ollama(prompt: str) -> str:
        try:
            response = requests.post(OLLAMA_API, json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            })
            response.raise_for_status()
            return response.json()['response'].strip()
        except Exception as e:
            print(f"Ollama error: {e}")
            return "Error"

    # --------------------------
    # BUILD PROMPT
    # --------------------------

    def build_prompt(text: str) -> str:
        if language == "en":
            return f"<s>[INST] Classify the following text into one SDG category from the list: [{subjects_str}]. Return only the most relevant category, with no further explanation. I want the answer to include ONLY THE NUMBER OF THE CATEGORY! Text: {text} [/INST]"
        elif language == "fr":
            return f"<s>[INST] Classez le texte suivant dans une seule catégorie des ODD parmi la liste : [{subjects_str}]. Retournez uniquement la catégorie la plus pertinente, sans aucune explication. Texte : {text} [/INST]"
        elif language == "ar":
            return f"<s>[INST] صنّف النص التالي ضمن فئة واحدة فقط من أهداف التنمية المستدامة من القائمة التالية: [{subjects_str}]. أرجع فقط الفئة الأكثر صلة دون أي تفسير إضافي. نص: {text} [/INST]"
        elif language == "es":
            return f"<s>[INST] Clasifica el siguiente texto en una sola categoría de los ODS de esta lista: [{subjects_str}]. Devuelve solo la categoría más relevante, sin ninguna explicación. Texto: {text} [/INST]"
        elif language == "ru":
            return f"<s>[INST] Классифицируйте следующий текст в одну категорию из списка ЦУР: [{subjects_str}]. Верните только одну наиболее подходящую категорию, без каких-либо объяснений. Текст: {text} [/INST]"
        elif language == "zh":
            return f"<s>[INST] 将以下文本分类为一个可持续发展目标类别，类别来自以下列表：[{subjects_str}]。仅返回最相关的一个类别，不要提供任何解释。文本：{text} [/INST]"

    # --------------------------
    # LOAD DATA
    # --------------------------

    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print("Skipping invalid line.")

    df = pd.DataFrame(data)

    # --------------------------
    # RUN CLASSIFICATION
    # --------------------------

    file_exists = False

    toGo = True

    for _, row in df.iterrows():

        if row.get("annotationId") == "11c54b4e-3aed-411f-bde9-8420bcf3e319":
            toGo = False

        if not toGo:
            continue

        text = re.sub(r'\n+', ' ', row[language + "_char_attack"]).strip()
        prompt = build_prompt(text)
        print(prompt)
        prediction = query_ollama(prompt)

        result = {
            "annotationId": row.get("annotationId", ""),
            "language": language,
            "sdg_categories": row.get("sdgsIds", []),
            "predicted": prediction
        }

        pd.DataFrame([result]).to_csv(csv_filename, mode='a', header=not file_exists, index=False)
        file_exists = True
