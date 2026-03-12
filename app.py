
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time, os, io, base64
from datetime import datetime

st.set_page_config(page_title="FruitScan AI", page_icon="🍎", layout="wide")

st.markdown('''<style>
@import url("https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap");
html,body,[class*="css"]{font-family:"DM Sans",sans-serif;background:#0e1a12;color:#e8f0e9}
[data-testid="stSidebar"]{background:linear-gradient(160deg,#0a1a0f,#142014);border-right:1px solid #2a4a2f}
.stButton>button{background:linear-gradient(135deg,#2a7030,#3a9040)!important;color:#e0ffe0!important;border:none!important;border-radius:10px!important;font-size:1rem!important}
div[data-testid="stFileUploadDropzone"]{background:#162d1a!important;border:2px dashed #3a6040!important;border-radius:12px!important}
</style>''', unsafe_allow_html=True)

NUM_CLASSES = 21
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
CHECKPOINT_DIR = '/content/drive/MyDrive/checkpoints'

CLASS_NAMES = ['apple','banana','broccoli','carrot','cucumber','date_plum','garlic',
    'grape','kiwi','lemon','mandarin','onion','orange','pear','persimmon',
    'pineapple','pitahaya','pomegranate','potato','quince','strawberry']

EMOJIS = {'apple':'🍎','banana':'🍌','broccoli':'🥦','carrot':'🥕','cucumber':'🥒',
    'date_plum':'🫐','garlic':'🧄','grape':'🍇','kiwi':'🥝','lemon':'🍋',
    'mandarin':'🍊','onion':'🧅','orange':'🍊','pear':'🍐','persimmon':'🍅',
    'pineapple':'🍍','pitahaya':'🐉','pomegranate':'🍎','potato':'🥔','quince':'🍋','strawberry':'🍓'}

PRICES = {'apple':250,'banana':120,'broccoli':400,'carrot':80,'cucumber':150,
    'date_plum':200,'garlic':100,'grape':350,'kiwi':180,'lemon':150,
    'mandarin':180,'onion':60,'orange':250,'pear':220,'persimmon':300,
    'pineapple':1200,'pitahaya':800,'pomegranate':500,'potato':55,'quince':200,'strawberry':400}

val_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

def build_resnet50(n):
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.4),nn.Linear(m.fc.in_features,512),nn.ReLU(),nn.Dropout(0.3),nn.Linear(512,n))
    return m
def build_efficientnet(n):
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.4),nn.Linear(m.classifier[1].in_features,512),nn.ReLU(),nn.Dropout(0.3),nn.Linear(512,n))
    return m
def build_vgg16(n):
    m = models.vgg16(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.5),nn.Linear(25088,1024),nn.ReLU(inplace=True),nn.Dropout(0.4),nn.Linear(1024,n))
    return m
def build_googlenet(n):
    m = models.googlenet(weights=None, aux_logits=True)
    m.aux1 = None
    m.aux2 = None
    m.fc = nn.Sequential(nn.Dropout(0.4),nn.Linear(m.fc.in_features,512),nn.ReLU(inplace=True),nn.Dropout(0.3),nn.Linear(512,n))
    return m
def build_alexnet(n):
    m = models.alexnet(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.5),nn.Linear(9216,1024),nn.ReLU(inplace=True),nn.Dropout(0.4),nn.Linear(1024,n))
    return m

MODELS = {
    "ResNet-50":       (build_resnet50,    "resnet50_best.pth"),
    "EfficientNet-B0": (build_efficientnet,"efficientnet_best.pth"),
    "VGG-16":          (build_vgg16,       "vggnet_best.pth"),
    "GoogLeNet":       (build_googlenet,   "googlenet_best.pth"),
    "AlexNet":         (build_alexnet,     "alexnet_best.pth"),
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource(show_spinner=False)
def load_model(name):
    builder, fname = MODELS[name]
    net = builder(NUM_CLASSES)
    path = os.path.join(CHECKPOINT_DIR, fname)
    if os.path.exists(path):
        ck = torch.load(path, map_location=DEVICE)
        net.load_state_dict(ck.get("model_state_dict", ck))
        ok = True
    else:
        ok = False
    return net.to(DEVICE).eval(), ok

def predict(img, model, k=5):
    t = val_tf(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(t), dim=1)[0].cpu()
    tp, ti = probs.topk(k)
    return [{"class":CLASS_NAMES[i],"prob":float(p),"price":PRICES.get(CLASS_NAMES[i],0)} for p,i in zip(tp,ti)]

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.markdown("### 🌿 FruitScan AI")
    st.markdown("---")
    st.markdown("**🤖 Выбери модель**")
    chosen = st.selectbox("", list(MODELS.keys()), label_visibility="collapsed")
    top_k = st.slider("Top-K предсказаний", 1, 5, 3)
    st.markdown("---")
    st.markdown("**📋 Все фрукты:**")
    for c in CLASS_NAMES:
        st.markdown(f"{EMOJIS.get(c,'🌿')} {c.replace('_',' ').title()}")
    if st.button("🗑️ Очистить историю"):
        st.session_state.history = []
        st.rerun()

st.markdown('<p style="font-family:Playfair Display,serif;font-size:2.8rem;font-weight:700;background:linear-gradient(135deg,#7dde7d,#ffd580);-webkit-background-clip:text;-webkit-text-fill-color:transparent">FruitScan AI 🍃</p>', unsafe_allow_html=True)
st.markdown("Загрузи фото фрукта — получи название, уверенность и цену")

with st.spinner(f"Загружаю {chosen}..."):
    net, loaded = load_model(chosen)

if not loaded:
    st.warning(f"⚠️ Чекпоинт не найден в {CHECKPOINT_DIR}. Проверь путь!")

col1, col2 = st.columns([1,1], gap="large")

with col1:
    st.markdown("### 📤 Загрузи изображение")
    uploaded = st.file_uploader("", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
    if uploaded:
        image = Image.open(uploaded)
        st.image(image, use_container_width=True)
        btn = st.button("🔍 Определить фрукт", use_container_width=True)
    else:
        st.markdown('<div style="text-align:center;padding:3rem;color:#4a7a4a;border:2px dashed #2a4a2f;border-radius:12px"><div style="font-size:3rem">🍎</div><div>Загрузи фото фрукта</div></div>', unsafe_allow_html=True)
        btn = False

with col2:
    st.markdown("### 🧠 Результат")
    if uploaded and btn:
        with st.spinner("Анализирую..."):
            t0 = time.time()
            results = predict(image, net, top_k)
            elapsed = time.time() - t0
        top = results[0]
        em = EMOJIS.get(top["class"], "🌿")
        st.markdown(f'<div style="background:linear-gradient(135deg,#162d1a,#1e3a22);border:1px solid #3a6040;border-radius:16px;padding:1.8rem;margin-top:1rem"><div style="font-size:3rem">{em}</div><div style="font-family:Playfair Display,serif;font-size:2rem;color:#a8f0a8;font-weight:700">{top["class"].replace("_"," ").title()}</div><div style="display:inline-block;background:linear-gradient(135deg,#2a6030,#3a8040);color:#c0f0a0;font-size:1.4rem;font-weight:700;padding:0.4rem 1.2rem;border-radius:50px;margin-top:0.5rem">₸ {top["price"]:,} / шт</div><div style="color:#8aaa8a;font-size:0.8rem;margin-top:1rem;text-transform:uppercase">Уверенность</div><div style="font-size:1.8rem;color:#7dde7d;font-weight:700">{top["prob"]*100:.1f}%</div><div style="font-size:0.75rem;color:#4a7a4a;margin-top:0.5rem">⚡ {elapsed*1000:.0f} ms · {chosen} · {DEVICE.upper()}</div></div>', unsafe_allow_html=True)
        if len(results) > 1:
            st.markdown("**Другие варианты:**")
            for r in results:
                st.markdown(f"{EMOJIS.get(r['class'],'🌿')} **{r['class'].replace('_',' ').title()}** — {r['prob']*100:.1f}% · ₸{r['price']:,}")
                st.progress(r['prob'])
        thumb = image.copy()
        thumb.thumbnail((120,120))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG")
        st.session_state.history.insert(0,{"fruit":top["class"],"model":chosen,"conf":top["prob"],"price":top["price"],"time":datetime.now().strftime("%H:%M:%S"),"thumb":base64.b64encode(buf.getvalue()).decode()})
    else:
        st.markdown('<div style="text-align:center;padding:4rem;color:#4a7a4a"><div style="font-size:2.5rem">🔬</div><div>Результат появится здесь</div></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 💰 Цены на фрукты")
cols = st.columns(7)
items = list(PRICES.items())
for i, (fruit, price) in enumerate(items):
    cols[i%7].markdown(f'<div style="background:#162d1a;border:1px solid #2a4a2f;border-radius:10px;padding:0.6rem;text-align:center;margin-bottom:0.5rem"><div style="font-size:1.4rem">{EMOJIS.get(fruit,"🌿")}</div><div style="font-size:0.7rem;color:#a0c8a0">{fruit.replace("_"," ").title()}</div><div style="font-size:0.85rem;color:#7dde7d;font-weight:600">₸{price}</div></div>', unsafe_allow_html=True)

if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕑 История определений")
    hcols = st.columns(4)
    for idx, entry in enumerate(st.session_state.history[:8]):
        hcols[idx%4].markdown(f'<div style="background:#162d1a;border:1px solid #2a4a2f;border-radius:12px;overflow:hidden;margin-bottom:0.5rem"><img src="data:image/jpeg;base64,{entry["thumb"]}" style="width:100%;height:90px;object-fit:cover"><div style="padding:0.6rem"><div style="color:#a8f0a8;font-weight:500">{EMOJIS.get(entry["fruit"],"🌿")} {entry["fruit"].replace("_"," ").title()}</div><div style="font-size:0.72rem;color:#6a8a6a">{entry["conf"]*100:.1f}% · ₸{entry["price"]:,}</div><div style="font-size:0.68rem;color:#4a6a4a">{entry["model"]} · {entry["time"]}</div></div></div>', unsafe_allow_html=True)
