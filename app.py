##1)Importation des bibliothèques
import streamlit as st
import speech_recognition as sr
from PIL import Image
import pytesseract
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyttsx3
import torch

# =========================
#2) Configuration générale
# =========================

torch.set_num_threads(2)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
st.set_page_config(page_title="Mini Google Traduction + Chatbot", layout="wide")

# =========================
#3) CSS ESTHÉTIQUE AVANCÉ
# =========================

st.markdown("""
<style>
/* Reset et base */
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', -apple-system, sans-serif; }

/* Header animé amélioré */
.title-box {
    background: linear-gradient(270deg, #ff0000, #ff9900, #ffee00, #00ff00, #00ffff, #0066ff, #9900ff, #ff0000);
    background-size: 1400% 1400%;
    animation: gradient 8s ease infinite, glow 2s ease-in-out infinite alternate;
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-weight: bold;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    position: relative;
    overflow: hidden;
}
.title-box::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%);
    animation: rotate 20s linear infinite;
}
@keyframes gradient { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
@keyframes glow { 0%{box-shadow:0 10px 30px rgba(0,0,0,0.4);} 100%{box-shadow:0 10px 50px rgba(255,100,100,0.6);} }
@keyframes rotate { 0%{transform:rotate(0deg);} 100%{transform:rotate(360deg);} }

/* Cartes stylées */
.card {
    background: linear-gradient(145deg, #ffffff, #f0f2f5);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.1), 0 5px 15px rgba(0,0,0,0.07);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    border: 1px solid rgba(255,255,255,0.8);
}
.card:hover {
    transform: translateY(-8px);
    box-shadow: 0 25px 50px rgba(0,0,0,0.15), 0 10px 25px rgba(0,0,0,0.1);
}

/* Boutons premium */
.stButton > button {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 50px;
    padding: 12px 30px;
    font-weight: 600;
    font-size: 14px;
    transition: all 0.3s ease;
    box-shadow: 0 8px 25px rgba(102,126,234,0.4);
    position: relative;
    overflow: hidden;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(102,126,234,0.6);
    background: linear-gradient(45deg, #764ba2 0%, #667eea 100%);
}
.stButton > button:active {
    transform: translateY(0);
}

/* Boutons secondaires (ex: Lire, Reset) */
.btn-secondary {
    background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
    box-shadow: 0 8px 25px rgba(245,87,108,0.4);
}
.btn-secondary:hover {
    box-shadow: 0 12px 35px rgba(245,87,108,0.6);
    background: linear-gradient(45deg, #f5576c 0%, #f093fb 100%);
}

/* Sidebar améliorée */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
}
section[data-testid="stSidebar"] .stTextArea > label, 
section[data-testid="stSidebar"] .stFileUploader > label {
    color: white !important;
    font-weight: 600;
}

/* Tabs stylées */
.stTabs [data-baseweb="tab-list"] {
    background: linear-gradient(145deg, #f0f2f5, #e2e8f0);
    border-radius: 15px;
    padding: 5px;
    gap: 5px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    padding: 12px 24px;
    transition: all 0.3s ease;
    font-weight: 600;
}
.stTabs [data-baseweb="tab"]:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    color: white;
    box-shadow: 0 10px 30px rgba(102,126,234,0.4);
}

/* Text areas avec style */
.stTextArea textarea, .stTextArea div[role="textbox"] {
    border-radius: 15px;
    border: 2px solid #e2e8f0;
    padding: 15px;
    transition: all 0.3s ease;
    box-shadow: inset 0 2px 10px rgba(0,0,0,0.05);
}
.stTextArea textarea:focus, .stTextArea div[role="textbox"]:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102,126,234,0.1), inset 0 2px 10px rgba(0,0,0,0.05);
}

/* Messages de succès/erreur stylés */
.stSuccess, .stError, .stInfo {
    border-radius: 15px;
    padding: 15px 20px;
    margin: 10px 0;
    box-shadow: 0 8px 25px rgba(0,0,0,0.1);
}

/* Historique listé */
.stMarkdown li { margin-bottom: 10px; }

/* Responsive */
@media (max-width: 768px) {
    .card { padding: 20px 15px; }
    .title-box { padding: 15px 10px; font-size: 1.2em; }
}
</style>
""", unsafe_allow_html=True)

# =========================
# 4)SESSION STATE
# =========================

if "final_text" not in st.session_state:
    st.session_state["final_text"] = ""
if "translated_text" not in st.session_state:
    st.session_state["translated_text"] = ""
if "history" not in st.session_state:
    st.session_state["history"] = []

# =========================
# 5)HEADER ANIMÉ
# =========================

st.markdown("""
<div class="title-box">
<h1>🌍 Mini Google Traduction + Chatbot IA OFFLINE</h1>
<p style="font-size: 1.1em; margin-top: 10px;">Traduction précise • Reconnaissance vocale • OCR • Chat IA</p>
</div>
""", unsafe_allow_html=True)

# =========================
#6) Gestion des langues
# =========================

langues = {
    "Français": "fr",
    "Anglais": "en",
    "Arabe": "ar",
    "Chinois": "zh",
    "Japonais": "ja"
}

SUPPORTED_DIRECT = {
    ("fr", "en"), ("en", "fr"),
    ("fr", "ar"), ("ar", "fr"),
    ("en", "ar"), ("ar", "en"),
    ("en", "zh"), ("zh", "en"),
    ("ja", "en"), ("en", "ja")
}

# =========================
# 7)Synthèse vocale (Text To Speech)
# =========================

def read_text_offline(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

@st.cache_resource       #8) Chargement des modèles Marian
def load_marian_model(src, tgt):
    name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tokenizer = MarianTokenizer.from_pretrained(name, local_files_only=True)
    model = MarianMTModel.from_pretrained(name, local_files_only=True)
    return tokenizer, model

def translate(text, src, tgt):       ##9) Fonction de traduction
    if (src, tgt) in SUPPORTED_DIRECT:
        tokenizer, model = load_marian_model(src, tgt)
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        output = model.generate(**inputs)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    else:
        t1, m1 = load_marian_model(src, "en")
        mid = t1.decode(m1.generate(**t1(text, return_tensors="pt"))[0], skip_special_tokens=True)
        t2, m2 = load_marian_model("en", tgt)
        return t2.decode(m2.generate(**t2(mid, return_tensors="pt"))[0], skip_special_tokens=True)

# =========================
# 10)SIDEBAR (avec classes CSS)
# =========================

with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("🌟 Options")
    st.markdown('</div>', unsafe_allow_html=True)

    option = st.radio("Choisissez la saisie", ["Texte", "Fichier TXT", "Audio", "Image"])

    # -------- TEXTE --------
    if option == "Texte":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        text_input = st.text_area("Entrez texte")
        st.markdown('</div>', unsafe_allow_html=True)
        if text_input.strip():
            st.session_state["final_text"] = text_input
            try:
                st.info(f"Langue: {detect(text_input)}")
            except:
                pass
        if st.button("🔊 Lire texte", key="read_text", help="Écouter le texte saisi"):
            read_text_offline(st.session_state["final_text"])

    # -------- TXT --------
    elif option == "Fichier TXT":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Importer TXT", type=["txt"])
        st.markdown('</div>', unsafe_allow_html=True)
        if uploaded_file:
            content = uploaded_file.read().decode("utf-8")
            st.session_state["final_text"] = content
            st.success("TXT chargé")
            if st.button("🔊 Lire fichier", key="read_txt"):
                read_text_offline(content)

    # -------- AUDIO --------
    elif option == "Audio":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.button("🎙 Démarrer micro", key="audio_btn"):
            r = sr.Recognizer()
            with sr.Microphone() as source:
                audio_data = r.listen(source, phrase_time_limit=8)
                try:
                    text_audio = r.recognize_google(audio_data)
                    st.session_state["final_text"] = text_audio
                    st.success("Audio reconnu")
                    read_text_offline(text_audio)
                except:
                    st.error("Erreur audio")
        st.markdown('</div>', unsafe_allow_html=True)

    # -------- IMAGE OCR --------
    elif option == "Image":
        st.markdown('<div class="card">', unsafe_allow_html=True)
        image_file = st.file_uploader("Importer image", type=["png","jpg","jpeg"])
        st.markdown('</div>', unsafe_allow_html=True)
        if image_file:
            img = Image.open(image_file)
            text_ocr = pytesseract.image_to_string(img)
            st.session_state["final_text"] = text_ocr
            st.success("OCR OK")
            if st.button("🔊 Lire OCR", key="read_ocr"):
                read_text_offline(text_ocr)

# =========================
# TABS
# =========================

tab1, tab2 = st.tabs(["🌐 Traduction", "💬 Chatbot IA"])

# =========================
# 11)ONGLET TRADUCTION
# =========================

with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📥 Texte source")
    st.text_area("Contenu", value=st.session_state["final_text"], height=150, key="source_text")
    st.markdown('</div>', unsafe_allow_html=True)

    col_lang, _ = st.columns([1, 3])
    with col_lang:
        st.markdown('<div class="card" style="padding:15px;">', unsafe_allow_html=True)
        source_lang = st.selectbox("Langue source", list(langues.keys()))
        target_lang = st.selectbox("Langue cible", list(langues.keys()), index=1)
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔁 Traduire", help="Lancer la traduction"):
        src = langues[source_lang]
        tgt = langues[target_lang]
        if src != tgt and st.session_state["final_text"]:
            with st.spinner("Traduction en cours..."):
                result = translate(st.session_state["final_text"], src, tgt)
                st.session_state["translated_text"] = result
                st.success("✅ Traduction terminée !")

    if st.session_state["translated_text"]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.text_area("Résultat traduit", st.session_state["translated_text"], height=150, key="translated_text")
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔊 Lire traduction", key="read_trans", help="Écouter la traduction"):
                read_text_offline(st.session_state["translated_text"])
        with col2:
            if st.button("➕ Ajouter historique", key="add_hist"):
                st.session_state["history"].append(st.session_state["translated_text"])
                st.success("Ajouté à l'historique")
        with col3:
            if st.button("📜 Afficher historique", key="show_hist"):
                if st.session_state["history"]:
                    st.markdown("### Historique des traductions")
                    for i, h in enumerate(st.session_state["history"][-10:], 1):  # Derniers 10
                        st.markdown(f"**{i}.** {h}")
                else:
                    st.info("Historique vide")

        if st.session_state["history"]:
            export_text = "\n\n".join(st.session_state["history"])
            st.download_button("⬇ Export TXT Historique", export_text, file_name="traductions_history.txt")

# =========================
# ONGLET CHATBOT
# =========================

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💬 Chatbot IA OFFLINE")
    st.markdown('</div>', unsafe_allow_html=True)

    @st.cache_resource
    def load_chatbot():
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small", local_files_only=True)
        return tokenizer, model

    tokenizer_chat, model_chat = load_chatbot()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    col_chat, col_reset = st.columns([4, 1])
    with col_chat:
        user_input = st.text_input("Votre message", key="chat_input")
    with col_reset:
        if st.button("🧹 Reset", key="reset_chat", help="Effacer la conversation"):
            st.session_state.chat_history = []
            st.rerun()

    if st.button("Envoyer", key="send_chat") and user_input:
        with st.spinner("IA réfléchit..."):
            new_input_ids = tokenizer_chat.encode(user_input + tokenizer_chat.eos_token, return_tensors='pt')
            bot_input_ids = torch.cat(st.session_state.chat_history + [new_input_ids], dim=-1) if st.session_state.chat_history else new_input_ids
            output_ids = model_chat.generate(bot_input_ids, max_length=700, pad_token_id=tokenizer_chat.eos_token_id)
            response = tokenizer_chat.decode(output_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            st.session_state.chat_history.append(new_input_ids)
            st.session_state.chat_history.append(output_ids[:, bot_input_ids.shape[-1]:])
            st.markdown(f"**🤖 IA :** {response}")
