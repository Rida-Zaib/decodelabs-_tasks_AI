import re

# ---- KNOWLEDGE BASE: one dictionary per language ----
# Each entry: trigger phrase -> response, all in that language.
RESPONSES = {
    "en": {
        "hello": "Hi there! How can I help you today?",
        "hi": "Hello! What can I do for you?",
        "how are you": "I'm just a program, but I'm running great! How about you?",
        "what is your name": "I'm ChatBot, a simple rule-based assistant built for Project 1.",
        "help": "You can say: hello, how are you, what is your name, or bye.",
    },
    "ur_roman": {  # Roman Urdu
        "salam": "Walaikum Assalam! Aap kaisay hain?",
        "assalam o alaikum": "Walaikum Assalam! Kya haal hai?",
        "kaisay ho": "Main theek hun, shukriya! Aap sunayen?",
        "kaise ho": "Main theek hun, shukriya! Aap sunayen?",
        "tumhara naam kya hai": "Mera naam ChatBot hai, main Project 1 ke liye bana hun.",
        "madad": "Aap keh saktay hain: salam, kaisay ho, tumhara naam kya hai, ya khuda hafiz.",
    },
    "ur": {  # Urdu script
        "سلام": "وعلیکم السلام! آج میں آپ کی کیا مدد کر سکتا ہوں؟",
        "کیسے ہو": "میں صرف ایک پروگرام ہوں، لیکن بہت اچھا چل رہا ہوں! آپ کیسے ہیں؟",
        "تمہارا نام کیا ہے": "میرا نام چیٹ بوٹ ہے، پراجیکٹ 1 کے لیے بنایا گیا ہوں۔",
    },
    "hi_roman": {  # Hinglish
        "namaste": "Namaste! Main aapki kaise madad kar sakta hoon?",
        "kaise ho": "Main bas ek program hoon, lekin badhiya chal raha hoon! Aap kaise hain?",
        "tumhara naam kya hai": "Mera naam ChatBot hai, Project 1 ke liye banaya gaya hoon.",
    },
    "hi": {  # Hindi script
        "नमस्ते": "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?",
        "कैसे हो": "मैं सिर्फ एक प्रोग्राम हूँ, लेकिन बहुत बढ़िया चल रहा हूँ! आप कैसे हैं?",
        "तुम्हारा नाम क्या है": "मेरा नाम चैटबॉट है, मुझे प्रोजेक्ट 1 के लिए बनाया गया है।",
    },
    "ar": {
        "مرحبا": "أهلاً بك! كيف يمكنني مساعدتك اليوم؟",
        "سلام": "أهلاً بك! كيف يمكنني مساعدتك اليوم؟",
        "كيف حالك": "أنا مجرد برنامج، لكن كل شيء يعمل بشكل رائع! كيف حالك أنت؟",
        "ما اسمك": "اسمي شات بوت، تم إنشائي من أجل المشروع الأول.",
    },
    "fa": {
        "سلام": "سلام! چطور می‌توانم به شما کمک کنم؟",
        "حالت چطوره": "من فقط یک برنامه هستم، اما عالی کار می‌کنم! شما چطورید؟",
        "اسمت چیه": "اسم من چت‌بات است، برای پروژه ۱ ساخته شده‌ام.",
    },
    "fr": {
        "bonjour": "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
        "salut": "Salut ! Que puis-je faire pour vous ?",
        "comment vas-tu": "Je ne suis qu'un programme, mais tout va bien ! Et toi ?",
        "quel est ton nom": "Je m'appelle ChatBot, créé pour le Projet 1.",
    },
    "es": {
        "hola": "¡Hola! ¿Cómo puedo ayudarte hoy?",
        "como estas": "Solo soy un programa, ¡pero funciono genial! ¿Y tú?",
        "cual es tu nombre": "Me llamo ChatBot, creado para el Proyecto 1.",
    },
    "de": {
        "hallo": "Hallo! Wie kann ich dir heute helfen?",
        "wie geht es dir": "Ich bin nur ein Programm, aber es läuft super! Und dir?",
        "wie heisst du": "Ich heiße ChatBot, gebaut für Projekt 1.",
    },
    "it": {
        "ciao": "Ciao! Come posso aiutarti oggi?",
        "come stai": "Sono solo un programma, ma va tutto alla grande! E tu?",
        "come ti chiami": "Mi chiamo ChatBot, creato per il Progetto 1.",
    },
    "pt": {
        "ola": "Olá! Como posso te ajudar hoje?",
        "como vai": "Sou apenas um programa, mas estou ótimo! E você?",
        "qual e o seu nome": "Meu nome é ChatBot, criado para o Projeto 1.",
    },
    "ru": {
        "привет": "Привет! Чем я могу тебе помочь?",
        "как дела": "Я просто программа, но всё отлично! А у тебя?",
        "как тебя зовут": "Меня зовут ЧатБот, я создан для Проекта 1.",
    },
    "tr": {
        "merhaba": "Merhaba! Bugün sana nasıl yardımcı olabilirim?",
        "nasilsin": "Ben sadece bir programım ama harika çalışıyorum! Ya sen?",
        "adin ne": "Benim adım ChatBot, Proje 1 için oluşturuldum.",
    },
    "zh": {
        "你好": "你好!我今天能帮你什么?",
        "你怎么样": "我只是一个程序,不过运行得很好!你呢?",
        "你叫什么名字": "我叫聊天机器人,是为项目一制作的。",
    },
    "ja": {
        "こんにちは": "こんにちは!今日はどんなお手伝いができますか?",
        "元気ですか": "私はただのプログラムですが、絶好調です!あなたは?",
        "名前は何ですか": "私の名前はチャットボットです、プロジェクト1のために作られました。",
    },
    "ko": {
        "안녕하세요": "안녕하세요! 오늘 어떻게 도와드릴까요?",
        "잘 지내요": "저는 그냥 프로그램이지만 아주 잘 작동하고 있어요! 당신은요?",
        "이름이 뭐에요": "제 이름은 챗봇이고, 프로젝트 1을 위해 만들어졌어요.",
    },
    "bn": {
        "হ্যালো": "হ্যালো! আজ আমি আপনাকে কীভাবে সাহায্য করতে পারি?",
        "কেমন আছো": "আমি শুধু একটি প্রোগ্রাম, কিন্তু দারুণ চলছি! আপনি কেমন আছেন?",
        "তোমার নাম কি": "আমার নাম চ্যাটবট, প্রজেক্ট ১-এর জন্য তৈরি।",
    },
    "id": {
        "halo": "Halo! Ada yang bisa saya bantu hari ini?",
        "apa kabar": "Saya hanya program, tapi berjalan lancar! Bagaimana denganmu?",
        "siapa namamu": "Nama saya ChatBot, dibuat untuk Proyek 1.",
    },
    "sw": {
        "habari": "Habari! Ninawezaje kukusaidia leo?",
        "u hali gani": "Mimi ni programu tu, lakini nafanya vizuri sana! Wewe je?",
        "jina lako ni nani": "Jina langu ni ChatBot, nilijengwa kwa ajili ya Mradi 1.",
    },
}

# ---- exit words per language ----
EXIT_WORDS = {
    "en": {"bye", "exit", "quit"},
    "ur_roman": {"khuda hafiz", "allah hafiz", "bye"},
    "ur": {"خدا حافظ"},
    "hi_roman": {"alvida", "bye"},
    "hi": {"अलविदा"},
    "ar": {"مع السلامة", "وداعا"},
    "fa": {"خداحافظ"},
    "fr": {"au revoir", "quitter"},
    "es": {"adios", "chao"},
    "de": {"tschuss", "tschüss"},
    "it": {"arrivederci", "addio"},
    "pt": {"tchau", "adeus"},
    "ru": {"пока", "до свидания"},
    "tr": {"hoscakal", "güle güle"},
    "zh": {"再见"},
    "ja": {"さようなら"},
    "ko": {"안녕히 가세요", "잘가"},
    "bn": {"বিদায়"},
    "id": {"selamat tinggal", "dadah"},
    "sw": {"kwaheri"},
}

EXIT_REPLY = {
    "en": "Goodbye! Have a great day!", "ur_roman": "Khuda Hafiz! Apna khayal rakhna.",
    "ur": "خدا حافظ! اپنا خیال رکھیں۔", "hi_roman": "Alvida! Aapka din shubh ho.",
    "hi": "अलविदा! आपका दिन शुभ हो।", "ar": "مع السلامة! أتمنى لك يوماً سعيداً.",
    "fa": "خداحافظ! روز خوبی داشته باشید.", "fr": "Au revoir ! Passe une bonne journée !",
    "es": "¡Adiós! Que tengas un buen día.", "de": "Tschüss! Einen schönen Tag noch!",
    "it": "Arrivederci! Buona giornata!", "pt": "Tchau! Tenha um ótimo dia!",
    "ru": "Пока! Хорошего дня!", "tr": "Hoşça kal! İyi günler!",
    "zh": "再见!祝你有美好的一天!", "ja": "さようなら!良い一日を!",
    "ko": "안녕히 가세요! 좋은 하루 되세요!", "bn": "বিদায়! আপনার দিন শুভ হোক।",
    "id": "Selamat tinggal! Semoga harimu menyenangkan!", "sw": "Kwaheri! Uwe na siku njema!",
}

FALLBACK = {
    "en": "I do not understand. Type 'help' to see what I can respond to.",
    "ur_roman": "Mujhe samajh nahi aya. 'madad' likh kar dekhen.",
    "ur": "مجھے سمجھ نہیں آیا۔", "hi_roman": "Mujhe samajh nahi aaya. 'madad' likhein.",
    "hi": "मुझे समझ नहीं आया। 'मदद' लिखें।", "ar": "لم أفهم. اكتب 'مساعدة'.",
    "fa": "متوجه نشدم.", "fr": "Je ne comprends pas. Tapez 'aide'.",
    "es": "No entiendo. Escribe 'ayuda'.", "de": "Das verstehe ich nicht. Schreib 'hilfe'.",
    "it": "Non capisco. Scrivi 'aiuto'.", "pt": "Não entendi. Digite 'ajuda'.",
    "ru": "Я не понимаю. Напиши 'помощь'.", "tr": "Anlamadım. 'yardım' yaz.",
    "zh": "我不明白。输入'帮助'看看我能回答什么。", "ja": "わかりません。「ヘルプ」と入力してください。",
    "ko": "이해하지 못했어요. '도움말'이라고 입력해보세요.", "bn": "আমি বুঝতে পারিনি।",
    "id": "Saya tidak mengerti. Ketik 'bantuan'.", "sw": "Sielewi. Andika 'msaada'.",
}

# ---- keyword signatures for languages that share the Latin alphabet ----
# (script-based languages below don't need this -- Unicode range is enough)
LATIN_SIGNATURES = {
    "ur_roman": {"salam", "kaisay", "kaise", "hain", "kya", "shukriya", "khuda", "hafiz", "madad"},
    "hi_roman": {"namaste", "kaise", "aap", "hoon", "hain", "alvida"},
    "fr": {"bonjour", "salut", "comment", "merci", "vas", "tu", "es"},
    "es": {"hola", "como", "estas", "gracias", "adios", "cual"},
    "de": {"hallo", "wie", "geht", "danke", "tschuss", "heisst"},
    "it": {"ciao", "come", "stai", "grazie", "chiami"},
    "pt": {"ola", "como", "vai", "obrigado", "tchau", "nome"},
    "tr": {"merhaba", "nasilsin", "tesekkur", "hoscakal", "adin"},
    "id": {"halo", "apa", "kabar", "terima", "kasih", "siapa", "namamu"},
    "sw": {"habari", "asante", "kwaheri", "jambo", "jina"},
}

# ---- Unicode ranges for script-based languages (100% deterministic) ----
SCRIPT_RANGES = [
    ("hi", re.compile(r"[\u0900-\u097F]")),   # Devanagari
    ("bn", re.compile(r"[\u0980-\u09FF]")),   # Bengali
    ("ru", re.compile(r"[\u0400-\u04FF]")),   # Cyrillic
    ("ja", re.compile(r"[\u3040-\u30FF]")),   # Hiragana / Katakana (check before CJK)
    ("ko", re.compile(r"[\uAC00-\uD7A3]")),   # Hangul
    ("zh", re.compile(r"[\u4E00-\u9FFF]")),   # CJK Unified Ideographs
    ("ARABIC_BLOCK", re.compile(r"[\u0600-\u06FF]")),  # Arabic / Urdu / Persian share this block
]

URDU_ONLY_CHARS = set("ٹڈڑںے")   # letters unique to Urdu, not used in Arabic/Persian
PERSIAN_ONLY_CHARS = set("پچژگ")  # letters unique to Persian (not standard Arabic)


def detect_language(raw_text: str) -> str:
    """Rule-based language detection: script (Unicode range) first,
    then keyword matching for languages that share the Latin alphabet."""
    for lang, pattern in SCRIPT_RANGES:
        if pattern.search(raw_text):
            if lang == "ARABIC_BLOCK":
                if any(ch in URDU_ONLY_CHARS for ch in raw_text):
                    return "ur"
                if any(ch in PERSIAN_ONLY_CHARS for ch in raw_text):
                    return "fa"
                return "ar"
            return lang

    words = set(raw_text.lower().split())
    best_lang, best_score = "en", 0
    for lang, signature in LATIN_SIGNATURES.items():
        score = len(words & signature)
        if score > best_score:
            best_lang, best_score = lang, score
    return best_lang


def get_response(clean_input: str, lang: str) -> str:
    return RESPONSES.get(lang, {}).get(clean_input, FALLBACK.get(lang, FALLBACK["en"]))


def main():
    print("ChatBot: Hello! I understand 20 languages, including Roman Urdu. Type 'bye' to exit.")

    while True:
        raw_input_text = input("You: ")
        clean_input = raw_input_text.lower().strip()

        lang = detect_language(raw_input_text)

        if clean_input in EXIT_WORDS.get(lang, set()) or raw_input_text.strip() in EXIT_WORDS.get(lang, set()):
            print(f"ChatBot: {EXIT_REPLY.get(lang, EXIT_REPLY['en'])}")
            break

        # script-based languages keep original casing (e.g. Urdu/Arabic have no case)
        lookup_key = raw_input_text.strip() if lang not in ("en", "ur_roman", "hi_roman", "fr", "es", "de", "it", "pt", "tr", "id", "sw") else clean_input
        reply = get_response(lookup_key, lang)
        print(f"ChatBot: {reply}")


if __name__ == "__main__":
    main()
