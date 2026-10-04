"""RIPEN LAB · Streamlit interface for banana ripeness YOLO models."""
from __future__ import annotations

import html
import io
import json
import tempfile
import base64
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "YOLOV26" / "weights" / "best.pt"
THAI = ["ยังไม่สุก", "เริ่มสุก", "สุกพร้อมใช้", "สุกงอม"]
NAMES = ["unripe", "semi_ripe", "ripe", "overripe"]
COLORS = ["#65C889", "#C9DA5B", "#FFD84A", "#B47E5C"]


@st.cache_data(show_spinner=False)
def banana_photo_data() -> list[str]:
    files = ["banana-unripe.png", "banana-semi-ripe.png", "banana-ripe.png", "banana-overripe.png"]
    return [base64.b64encode((ROOT / "assets" / name).read_bytes()).decode("ascii") for name in files]


def banana_animation() -> str:
    frames = []
    for i, data in enumerate(banana_photo_data()):
        frames.append(f'<div class="ripeness-frame frame-{i}"><img src="data:image/png;base64,{data}" alt="" decoding="sync"/></div>')
    return '<div class="ripeness-cycle" aria-hidden="true">' + ''.join(frames) + '</div>'

st.set_page_config(page_title="RIPEN LAB · Banana Intelligence", page_icon="🍌", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Noto+Sans+Thai:wght@400;500;600;700;800&display=swap');
html,body,[class*=css],[data-testid=stApp]{font-family:'DM Sans','Noto Sans Thai',sans-serif}
[data-testid=stApp]{color:#202324;background-color:#f4f6eb;background-image:radial-gradient(ellipse at 100% 0%,#f9e9ab 0%,transparent 34%),radial-gradient(ellipse at 0% 66%,#dcebd3 0%,transparent 39%),linear-gradient(145deg,#faf9f1 3%,#edf3e8 64%,#f7f2e4 100%);background-attachment:fixed}
.block-container:before{content:'';position:fixed;inset:0;pointer-events:none;background-image:radial-gradient(#96aa8f55 1px,transparent 1px);background-size:22px 22px;mask-image:linear-gradient(90deg,#0009,transparent 18%,transparent 82%,#0008);z-index:0}
.block-container>div{position:relative;z-index:1}
.block-container{max-width:1360px;padding:1.4rem 2.5rem 5rem}
#MainMenu,footer{visibility:hidden}
[data-testid=stHeader]{background:transparent}
h1,h2,h3{color:#202324!important;letter-spacing:-.035em}
.site-head{display:flex;align-items:center;justify-content:space-between;padding:.85rem 1.1rem;background:#ffffffc9;backdrop-filter:blur(12px);border:1px solid #fff;border-radius:18px;box-shadow:0 8px 30px #46644712;margin-bottom:1.1rem}
.logo{font-size:1.34rem;font-weight:800;letter-spacing:-.045em;color:#242428}.logo span{background:#ffd84a;padding:.17rem .42rem;border-radius:8px;margin-right:.45rem}
.head-note{font-size:.74rem;font-weight:800;letter-spacing:.14em;color:#7b7c7b}
.head-right{display:flex;align-items:center;gap:.85rem}.head-status{display:inline-flex;align-items:center;gap:.4rem;border-radius:99px;background:#e7efe7;color:#286b47;padding:.35rem .7rem;font-size:.76rem;font-weight:800}.head-status.waiting{background:#fff2cb;color:#875f12}
div[data-testid=stRadio] > label{display:none}
div[data-testid=stRadio] div[role=radiogroup]{display:flex;gap:.4rem;flex-wrap:wrap;background:#eae9e4;border-radius:14px;padding:.32rem;width:max-content;max-width:100%;margin-bottom:1.2rem}
div[data-testid=stRadio] div[role=radiogroup] label{border-radius:10px;padding:.44rem 1rem;margin:0;background:transparent;font-weight:700;color:#464951}
div[data-testid=stRadio] div[role=radiogroup] label:has(input:checked){background:#25272c;color:white;box-shadow:0 3px 8px #25272c22}
div[data-testid=stRadio] div[role=radiogroup] label p{font-size:.88rem}
div[data-testid=stRadio] div[role=radiogroup] label:has(input:checked) p{color:white}
div[data-testid=stRadio] input{display:none}
.hero{background:radial-gradient(circle at 12% 4%,#39534677,transparent 36%),linear-gradient(108deg,#1c3028 0%,#20382e 57%,#34422c 100%);border:1px solid #63715e55;border-radius:29px;min-height:366px;padding:2.55rem 2.8rem;color:#fff;position:relative;overflow:hidden;margin-bottom:1.3rem;box-shadow:0 24px 42px #233b292a}
.hero:before{content:'';position:absolute;width:480px;height:480px;right:-65px;top:-190px;border-radius:50%;background:#ffdd50;box-shadow:0 0 95px #ffe47355;animation:haloCycle 20s linear infinite}
.hero:after{content:'';position:absolute;inset:0;background:repeating-radial-gradient(circle at 79% 37%,transparent 0 39px,#ffffff13 40px 41px);pointer-events:none;z-index:0}
.hero-copy{position:relative;z-index:2;max-width:630px;padding-top:.55rem}.hero .kicker{font-size:.74rem;letter-spacing:.18em;font-weight:800;color:#ffd84a}
.hero h1{font-size:clamp(2.4rem,4.1vw,4.3rem);line-height:1.1;color:white!important;letter-spacing:-.055em;margin:.6rem 0 .7rem}
.hero p{font-size:1.01rem;color:#cfd0d0;line-height:1.7;margin:0;max-width:530px}
.hero-detail{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1.6rem}.hero-detail span{color:#edf4e9;background:#ffffff1d;border:1px solid #ffffff2e;border-radius:99px;padding:.42rem .75rem;font-size:.7rem;letter-spacing:.08em;font-weight:800;backdrop-filter:blur(5px)}
.ripeness-cycle{position:absolute;right:0;top:0;width:44%;height:100%;pointer-events:none;z-index:1}
.ripeness-frame{position:absolute;inset:0;opacity:0}
.ripeness-frame img{position:absolute;width:110%;height:100%;object-fit:contain;right:-5%;top:3%;filter:drop-shadow(0 22px 17px #15171785);transform:rotate(-11deg)}
.frame-0{opacity:1}
.frame-1{animation:stageSemi 20s linear infinite}
.frame-2{animation:stageRipe 20s linear infinite}
.frame-3{animation:stageOverripe 20s linear infinite}
@keyframes stageSemi{0%,4%{opacity:0}25%,45%{opacity:1}50%,100%{opacity:0}}
@keyframes stageRipe{0%,25%{opacity:0}45%,70%{opacity:1}75%,100%{opacity:0}}
@keyframes stageOverripe{0%,45%{opacity:0}70%,85%{opacity:1}100%{opacity:0}}
@keyframes haloCycle{0%,100%{background:#b7e287}25%{background:#e9ea80}50%{background:#ffdd50}75%{background:#cda074}}
.strip{display:grid;grid-template-columns:1.1fr .9fr .9fr;gap:.8rem;margin:.1rem 0 2rem}
.strip-card{background:#fff;border:1px solid #e2e2db;border-radius:16px;padding:1rem 1.2rem;min-height:80px}
.strip-card b{display:block;font-size:.9rem}.strip-card small{display:block;color:#7d807d;font-size:.77rem;margin-top:.25rem}
.strip-card.yellow{background:#ffec9a;border-color:#ffec9a}
.work-aside{background:radial-gradient(circle at 92% 10%,#4c715b88,transparent 42%),linear-gradient(135deg,#1e342b,#243d32);color:#fff;border-radius:24px;padding:1.5rem;min-height:340px;box-shadow:0 15px 28px #20272718;position:relative;overflow:hidden}
.work-aside:before{content:'';position:absolute;width:185px;height:185px;border:1px solid #ffffff16;border-radius:50%;right:-82px;bottom:-95px;box-shadow:0 0 0 32px #ffffff08,0 0 0 65px #ffffff07;pointer-events:none}.work-aside>*{position:relative}
.work-aside .aside-kicker{font-size:.7rem;letter-spacing:.16em;font-weight:800;color:#ffdb62}.work-aside h3{color:#fff!important;margin:.8rem 0 .55rem;font-size:1.45rem}.work-aside p{font-size:.87rem;color:#c9cecb;line-height:1.7;margin:.25rem 0 1.15rem}
.work-aside .aside-rule{height:1px;background:#ffffff30;margin:1.25rem 0}.work-aside .aside-item{display:flex;align-items:center;gap:.65rem;margin:.77rem 0;color:#e2e8e1;font-size:.87rem}.work-aside .aside-no{color:#ffd84a;font-weight:800;font-size:.78rem}
.work-aside .model-chip{background:#ffffff17;color:#e6efe8;border:1px solid #ffffff2a;border-radius:10px;padding:.65rem .8rem;font-size:.8rem;line-height:1.5;margin-top:1rem;overflow-wrap:anywhere}
.stage-track{display:grid;grid-template-columns:repeat(4,1fr);gap:.7rem;margin:1rem 0 1.8rem}.stage-track .track-item{position:relative;background:linear-gradient(135deg,#fff 42%,color-mix(in srgb,var(--stage-color) 26%,white));border:1px solid #e3e3dc;border-radius:17px;padding:1rem 1.15rem;min-height:91px;box-shadow:0 7px 18px #536e4d0d;transition:transform .2s ease,box-shadow .2s ease}.stage-track .track-item:hover{transform:translateY(-3px);box-shadow:0 12px 23px #536e4d1c}.stage-track .track-item:before{content:'';display:block;width:34px;height:6px;border-radius:8px;background:var(--stage-color);margin-bottom:.67rem}.stage-track b{font-size:.9rem}.stage-track small{display:block;color:#788078;font-size:.72rem;letter-spacing:.05em}
.result-intro{border-radius:17px;background:#eaf1e7;border:1px solid #d1e4d0;padding:1rem 1.2rem;margin:1.4rem 0}.result-intro b{display:block;color:#28583e}.result-intro small{color:#697c6d}
.empty-state{background:#fff;border:1px solid #e2e4df;border-radius:22px;padding:2.2rem;margin:1.1rem 0 2rem;box-shadow:0 12px 28px #232b2709}.empty-state .empty-icon{display:inline-flex;align-items:center;justify-content:center;background:#fff2b9;border-radius:15px;width:46px;height:46px;font-size:1.4rem}.empty-state h3{margin:.95rem 0 .3rem;font-size:1.28rem}.empty-state p{color:#777e79;font-size:.9rem;margin:0;line-height:1.7}
.video-aside{background:radial-gradient(circle at 100% 0%,#d1e9c8,transparent 45%),#ecf2e9;color:#273d30;box-shadow:none;border:1px solid #d7e5d5}.video-aside .aside-kicker{color:#498456}.video-aside h3{color:#273d30!important}.video-aside p,.video-aside .aside-item{color:#54685b}.video-aside .aside-rule{background:#c8d9c9}.video-aside .aside-no{color:#328458}.video-aside .model-chip{background:#fff;color:#4b6854;border-color:#d5e1d4}
.split-title{display:flex;justify-content:space-between;align-items:end;margin:1.8rem 0 .9rem;padding-left:.9rem;border-left:4px solid #76a981}
.split-title h2{font-size:1.65rem;margin:0}.split-title span{color:#858789;font-size:.83rem}
.sub{color:#74777c;font-size:.91rem;margin:0 0 1rem}
.panel{background:white;border:1px solid #dedfda;border-radius:22px;padding:1.4rem;margin-bottom:1rem}
.how{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;margin:1rem 0}
.how>div{background:#fff;border:1px solid #e1e1dc;border-radius:17px;padding:1.25rem}
.how span{background:#25272c;color:#ffd84a;font-weight:800;padding:.35rem .65rem;border-radius:8px;font-size:.75rem}
.how b{display:block;margin:.85rem 0 .25rem}.how small{color:#777a7b;line-height:1.6}
.status{border-radius:17px;padding:1rem 1.2rem;display:flex;align-items:center;gap:1rem;margin:.8rem 0 1.25rem;background:#fff4d0;border:1px solid #efdb9a}
.status.good{background:#e7f3e8;border-color:#c9e7ce}.status strong{display:block;font-size:.91rem}.status small{color:#686c69;line-height:1.6}
.status .led{width:12px;height:12px;border-radius:50%;background:#e6a91d;flex:none}.status.good .led{background:#38ac69}
.class-row{display:grid;grid-template-columns:repeat(4,1fr);gap:.85rem;margin:1rem 0 1.9rem}
.class-card{border:1px solid #e2e2dd;border-radius:17px;padding:1.15rem;background:white}
.class-card .dot{height:16px;width:16px;border-radius:50%;display:block;margin-bottom:1rem}.class-card b{display:block}.class-card small{color:#86888a}
[data-testid=stFileUploader]{border:2px dashed #9ebba0;background:linear-gradient(135deg,#fff 65%,#f0f7eb);border-radius:22px;padding:1.1rem;min-height:166px;box-shadow:0 12px 30px #22272609}
[data-testid=stFileUploader] section{border:0;background:#f9f9f6;border-radius:13px}
[data-testid=stFileUploader] button{border-radius:9px}
[data-testid=stMetric]{background:#fff;border:1px solid #e0e0db;border-radius:15px;padding:1.1rem}
[data-testid=stMetricValue]{color:#25272c}
[data-testid=stVerticalBlockBorderWrapper]{border-radius:19px!important;border-color:#e1e1dc!important;background:#fff}
div.stButton>button[kind=primary]{background:#25272c;color:#fff;border:0;border-radius:12px;padding:.6rem 1.5rem;font-weight:800}
div.stButton>button[kind=primary]:hover{background:#44464d}
div.stDownloadButton>button{border-radius:11px;border:1px solid #bfc0b8;color:#22252a;font-weight:700}
div[data-testid=stExpander]{border:1px solid #e0e3dc;border-radius:15px;background:#fff}
.settings-note{font-size:.78rem;color:#797f79;margin:-.55rem 0 1.2rem}
.footer-note{font-size:.76rem;color:#888b89;border-top:1px solid #dcddd7;padding-top:1rem;margin-top:2.5rem}
@media(max-width:850px){.block-container{padding:1rem 1rem 3rem}.head-note{display:none}.hero{padding:1.8rem;min-height:350px}.hero-copy{max-width:100%}.hero h1{font-size:2.35rem}.hero p{max-width:78%;font-size:.88rem}.hero:before{right:-300px}.hero-detail{margin-top:1rem}.ripeness-cycle{width:62%;opacity:.7;right:-12%;top:50px}.strip,.how{grid-template-columns:1fr}.stage-track{grid-template-columns:repeat(2,1fr)}.class-row{grid-template-columns:repeat(2,1fr)}}
@media(max-width:520px){.site-head{padding:.75rem}.logo{font-size:1.08rem}.head-status{font-size:.67rem;padding:.3rem .48rem}.hero{min-height:410px;padding:1.5rem}.hero h1{font-size:2.15rem}.hero p{max-width:100%;font-size:.81rem}.hero-detail{max-width:72%}.hero-detail span{font-size:.57rem}.ripeness-cycle{width:85%;height:66%;right:-26%;top:34%;opacity:.6}.stage-track .track-item{min-height:77px;padding:.8rem}.split-title span{display:none}}
@media(prefers-reduced-motion:reduce){.ripeness-frame,.hero:before{animation:none!important}.ripeness-frame{opacity:0}.frame-2{opacity:1}}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="กำลังโหลดโมเดล...")
def load_model(path: str):
    from ultralytics import YOLO
    model = YOLO(path)
    names = model.names
    actual = [str(names[i]).lower().replace("-", "_").replace(" ", "_") for i in range(len(names))]
    if actual != NAMES:
        raise ValueError(f"คลาสโมเดลไม่ตรงกับงานกล้วย 4 ระดับ: {actual}")
    return model


def draw_result(image: Image.Image, boxes):
    image = image.convert("RGB").copy()
    painter = ImageDraw.Draw(image)
    font_path = next((p for p in (ROOT / "fonts").glob("*.ttf")), None)
    font = ImageFont.truetype(str(font_path), 19) if font_path else ImageFont.load_default()
    rows = []
    if boxes is None:
        return image, rows
    for box in boxes:
        cls = int(box.cls.item())
        if cls not in range(4):
            continue
        score = float(box.conf.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = COLORS[cls]
        painter.rectangle((x1, y1, x2, y2), outline=color, width=max(3, image.width // 250))
        label = f"{NAMES[cls]}  {score:.0%}"
        bb = painter.textbbox((0, 0), label, font=font)
        w, h = bb[2] + 18, bb[3] - bb[1] + 11
        top = max(0, y1 - h)
        painter.rectangle((x1, top, x1 + w, top + h), fill=color)
        painter.text((x1 + 9, top + 3), label, fill="#202324", font=font)
        rows.append({"ระดับความสุก": THAI[cls], "class": NAMES[cls], "ความมั่นใจ": round(score, 4), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return image, rows


def png_bytes(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def show_ripeness_counts(counts):
    """Display counts with native progress bars, without loading Arrow DLLs."""
    total = sum(int(counts.get(name, 0)) for name in NAMES)
    for label, name in zip(THAI, NAMES):
        count = int(counts.get(name, 0))
        share = count / total if total else 0.0
        st.progress(share, text=f"{label} · {count} · {share:.0%}")


def detection_table_html(rows):
    """Render the small detection summary without the Arrow table backend."""
    body = "".join(
        f"<tr><td>{html.escape(str(row['ระดับความสุก']))}</td>"
        f"<td>{float(row['ความมั่นใจ']):.1%}</td></tr>"
        for row in rows
    )
    return (
        '<table style="width:100%"><thead><tr><th>ระดับความสุก</th>'
        '<th>ความมั่นใจ</th></tr></thead><tbody>' + body + '</tbody></table>'
    )


def browser_video_bytes(source: Path) -> bytes:
    """Convert the OpenCV output to a browser-compatible H.264 MP4."""
    import imageio_ffmpeg

    destination = source.with_name("browser_result.mp4")
    process = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(source),
         "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart",
         str(destination)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    if process.returncode != 0:
        raise RuntimeError(f"แปลงวิดีโอเป็น H.264 ไม่สำเร็จ: {process.stderr[-1500:]}")
    return destination.read_bytes()


model_ready = MODEL_PATH.is_file()
status_class = "" if model_ready else " waiting"
status_text = "YOLO26s พร้อมใช้งาน" if model_ready else "รอโมเดล YOLO26s"
st.markdown(f'<div class="site-head"><div class="logo"><span>🍌</span> RIPEN LAB</div><div class="head-right"><div class="head-note">BANANA INTELLIGENCE / STUDIO 01</div><span class="head-status{status_class}">● {status_text}</span></div></div>', unsafe_allow_html=True)
page = st.radio("เมนู", ["สตูดิโอภาพ", "วิดีโอ", "ผลล่าสุด", "เกี่ยวกับโครงการ"], horizontal=True, label_visibility="collapsed")

with st.expander("⚙️ ตั้งค่าการวิเคราะห์ · YOLO26s"):
    setting_cols = st.columns([1.4, 1, 1], gap="large")
    setting_cols[0].markdown("**โมเดล:** YOLO26s · กล้วย 4 ระดับ" if model_ready else "**โมเดล:** ไม่พบ YOLOV26/weights/best.pt")
    confidence = setting_cols[1].slider("ความมั่นใจขั้นต่ำ", 0.05, 0.95, 0.25, 0.05)
    iou = setting_cols[2].slider("IoU ของกรอบซ้อน", 0.10, 0.90, 0.70, 0.05)
    st.caption("ค่าที่เลือกใช้กับภาพและวิดีโอ")

if page == "สตูดิโอภาพ":
    st.markdown('<div class="hero"><div class="hero-copy"><div class="kicker">IMAGE ANALYSIS · YOLO DETECTION</div><h1>จากภาพกล้วย<br>สู่ข้อมูลที่ใช้ได้จริง.</h1><p>ตรวจตำแหน่งและระดับความสุกในภาพเดียวหรือหลายภาพ แล้วดาวน์โหลดผลที่พร้อมนำไปใช้งาน</p><div class="hero-detail"><span>● 4 RIPENESS STAGES</span><span>IMAGE ANALYSIS</span><span>PNG + CSV EXPORT</span></div></div>'+banana_animation()+'</div>', unsafe_allow_html=True)
    st.markdown('<div class="stage-track"><div class="track-item" style="--stage-color:#66bf70"><b>ยังไม่สุก</b><small>UNRIPE</small></div><div class="track-item" style="--stage-color:#c4d35d"><b>เริ่มสุก</b><small>SEMI RIPE</small></div><div class="track-item" style="--stage-color:#ffd64a"><b>สุกพร้อมใช้</b><small>RIPE</small></div><div class="track-item" style="--stage-color:#af7957"><b>สุกงอม</b><small>OVERRIPE</small></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="split-title"><h2>พื้นที่วิเคราะห์ภาพ</h2><span>01 / IMAGE STUDIO</span></div>', unsafe_allow_html=True)
    guide_col, upload_col = st.columns([1, 1.65], gap="large")
    with upload_col:
        st.markdown("**เลือกภาพกล้วย**")
        st.caption("รองรับ JPG, PNG และ WEBP · เลือกหลายภาพได้พร้อมกัน")
        uploads = st.file_uploader("เลือกรูปกล้วยจากเครื่อง", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True, label_visibility="collapsed")
        if uploads:
            st.caption(f"เลือกแล้ว {len(uploads)} ภาพ")
            preview_cols = st.columns(min(3, len(uploads)))
            for col, upload in zip(preview_cols, uploads[:3]):
                try:
                    col.image(Image.open(upload).convert("RGB"), caption=upload.name, use_container_width=True)
                except Exception:
                    col.warning(f"อ่านภาพ {upload.name} ไม่ได้")
            if len(uploads) > 3:
                st.caption(f"และอีก {len(uploads) - 3} ภาพ")
        else:
            st.caption("ภาพที่ชัดและมีแสงเพียงพอจะช่วยให้ตรวจจับได้ดีขึ้น")
        analyze_clicked = st.button("เริ่มวิเคราะห์ภาพ →", type="primary", disabled=not uploads or not model_ready, use_container_width=True)
        if uploads and not model_ready:
            st.caption("เพิ่ม YOLOV26/weights/best.pt ก่อน แล้วปุ่มวิเคราะห์จะพร้อมใช้")
    with guide_col:
        model_info = "YOLO26s พร้อมใช้งาน · ปรับค่าได้ในเมนูตั้งค่าด้านบน" if model_ready else "ยังไม่มี YOLOV26/weights/best.pt ในโปรเจกต์"
        st.markdown('<div class="work-aside"><div class="aside-kicker">YOUR WORKFLOW</div><h3>ภาพของคุณ<br>พร้อมเข้าสู่การวิเคราะห์</h3><p>ดูตำแหน่งและระดับความสุกของกล้วย แล้วเก็บผลเป็นรูปภาพกับตารางข้อมูล</p><div class="aside-rule"></div><div class="aside-item"><span class="aside-no">01</span> เลือกภาพที่ต้องการ</div><div class="aside-item"><span class="aside-no">02</span> ตรวจด้วยโมเดล YOLO</div><div class="aside-item"><span class="aside-no">03</span> ดูและดาวน์โหลดผล</div><div class="model-chip">'+html.escape(model_info)+'</div></div>', unsafe_allow_html=True)
    if analyze_clicked:
        try:
            model = load_model(str(MODEL_PATH))
            results = []
            progress = st.progress(0, text="กำลังตรวจภาพ...")
            for i, upload in enumerate(uploads):
                with Image.open(upload) as original:
                    image = ImageOps.exif_transpose(original).convert("RGB")
                pred = model.predict(image, imgsz=640, conf=confidence, iou=iou, verbose=False)[0]
                rendered, rows = draw_result(image, pred.boxes)
                for row in rows:
                    row["ไฟล์"] = upload.name
                results.append({"name": upload.name, "image": png_bytes(rendered), "rows": rows})
                progress.progress((i + 1) / len(uploads), text=f"ตรวจภาพ {i+1}/{len(uploads)}")
            st.session_state["image_results"] = results
            st.success(f"วิเคราะห์เสร็จ {len(results)} ภาพ · ผลแสดงด้านล่างและในแท็บ ‘ผลล่าสุด’")
        except Exception as exc:
            st.error(f"วิเคราะห์ไม่สำเร็จ: {exc}")
    if st.session_state.get("image_results"):
        latest = st.session_state["image_results"]
        rows = [row for item in latest for row in item["rows"]]
        st.markdown('<div class="result-intro"><b>ผลการวิเคราะห์พร้อมแล้ว</b><small>ดูภาพที่ตรวจแล้วด้านล่าง หรือเปิดแท็บผลล่าสุดเพื่อดูสรุปทั้งหมด</small></div>', unsafe_allow_html=True)
        a, b = st.columns([1.6, 1])
        a.image(latest[0]["image"], caption=latest[0]["name"], use_container_width=True)
        b.metric("ภาพที่วิเคราะห์", len(latest))
        b.metric("กล้วยที่ตรวจพบ", len(rows))
        b.download_button("ดาวน์โหลดภาพนี้ PNG", latest[0]["image"], f"ripen_{Path(latest[0]['name']).stem}.png", "image/png", key="studio_download")
        if rows:
            st.download_button("ดาวน์โหลดผลทั้งหมด CSV", pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"), "ripen_results.csv", "text/csv", key="studio_csv")

elif page == "วิดีโอ":
    st.markdown('<div class="hero"><div class="hero-copy"><div class="kicker">VIDEO ANALYSIS · FRAME BY FRAME</div><h1>ดูความสุก<br>ตลอดทั้งคลิป.</h1><p>เลือกความถี่ในการตรวจเฟรม แล้วรับวิดีโอที่แสดงตำแหน่งกล้วยพร้อมบันทึกข้อมูลเป็น CSV</p><div class="hero-detail"><span>● MP4 INPUT</span><span>FRAME CONTROL</span><span>MP4 + CSV EXPORT</span></div></div>'+banana_animation()+'</div>', unsafe_allow_html=True)
    st.markdown('<div class="split-title"><h2>พื้นที่วิเคราะห์วิดีโอ</h2><span>02 / VIDEO STUDIO</span></div>', unsafe_allow_html=True)
    video_guide, video_col = st.columns([1, 1.65], gap="large")
    with video_col:
        st.markdown("**เลือกคลิปกล้วย**")
        st.caption("รองรับ MP4 · เลือกจำนวนเฟรมที่ต้องการตรวจ")
        video = st.file_uploader("อัปโหลดวิดีโอ MP4", type=["mp4"], key="video", label_visibility="collapsed")
        c1, c2 = st.columns(2)
        step = c1.select_slider("ตรวจทุกกี่เฟรม", options=[1, 2, 5, 10, 20], value=5)
        limit = int(c2.number_input("ตรวจสูงสุดกี่เฟรม", min_value=10, max_value=2000, value=300, step=10))
        analyze_video = st.button("เริ่มวิเคราะห์วิดีโอ →", type="primary", disabled=video is None or not model_ready, use_container_width=True)
        if video and not model_ready:
            st.caption("เพิ่ม YOLOV26/weights/best.pt ก่อน แล้วปุ่มวิเคราะห์จะพร้อมใช้")
    with video_guide:
        st.markdown('<div class="work-aside video-aside"><div class="aside-kicker">FRAME ANALYSIS</div><h3>เลือกเฟรม<br>แล้วดูความเปลี่ยนแปลง</h3><p>ระบบตรวจเฉพาะเฟรมตามช่วงที่เลือก และสร้างคลิปผลลัพธ์จากเฟรมเหล่านั้น</p><div class="aside-rule"></div><div class="aside-item"><span class="aside-no">01</span> อัปโหลด MP4</div><div class="aside-item"><span class="aside-no">02</span> กำหนดช่วงเฟรม</div><div class="aside-item"><span class="aside-no">03</span> ดาวน์โหลด MP4 และ CSV</div><div class="model-chip">จำนวนที่รายงานคือการพบในแต่ละเฟรม กล้วยลูกเดิมอาจถูกนับซ้ำ</div></div>', unsafe_allow_html=True)
    if analyze_video:
        try:
            import cv2
            model = load_model(str(MODEL_PATH))
            with tempfile.TemporaryDirectory() as temp:
                src = Path(temp) / "input.mp4"
                src.write_bytes(video.getvalue())
                cap = cv2.VideoCapture(str(src))
                if not cap.isOpened():
                    raise ValueError("เปิด MP4 ไม่สำเร็จ")
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                out = Path(temp) / "result.mp4"
                writer = None
                frame_i = sampled = 0
                records = []
                progress = st.progress(0, text="กำลังอ่านวิดีโอ...")
                try:
                    while sampled < limit:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        frame_i += 1
                        if (frame_i - 1) % step:
                            continue
                        sampled += 1
                        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        pred = model.predict(frame, conf=confidence, iou=iou, imgsz=640, verbose=False)[0]
                        rendered, boxes = draw_result(image, pred.boxes)
                        if writer is None:
                            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), max(1, fps / step), rendered.size)
                            if not writer.isOpened():
                                raise RuntimeError("ไม่สามารถบันทึก MP4 ผลลัพธ์ได้")
                        writer.write(cv2.cvtColor(np.array(rendered), cv2.COLOR_RGB2BGR))
                        records.extend({"วินาที": round((frame_i - 1) / fps, 2), "เฟรม": frame_i, **row} for row in boxes)
                        progress.progress(sampled / limit, text=f"ตรวจแล้ว {sampled} เฟรม")
                finally:
                    cap.release()
                    if writer:
                        writer.release()
                if sampled == 0:
                    raise ValueError("ไม่พบเฟรมที่อ่านได้")
                st.session_state["video_result"] = {"data": browser_video_bytes(out), "rows": records, "sampled": sampled}
                st.success(f"เสร็จแล้ว · ตรวจ {sampled} เฟรม · พบ {len(records)} ครั้ง")
        except Exception as exc:
            st.error(f"วิเคราะห์ไม่สำเร็จ: {exc}")
    if st.session_state.get("video_result"):
        result = st.session_state["video_result"]
        st.markdown('<div class="result-intro"><b>ผลวิดีโอพร้อมแล้ว</b><small>ดูคลิปที่ตรวจและดาวน์โหลดไฟล์ได้ด้านล่าง</small></div>', unsafe_allow_html=True)
        st.markdown('<div class="split-title"><h2>วิดีโอล่าสุด</h2><span>DOWNLOAD / EXPORT</span></div>', unsafe_allow_html=True)
        st.video(result["data"])
        c1, c2 = st.columns(2)
        c1.download_button("ดาวน์โหลด MP4", result["data"], "ripen_video_result.mp4", "video/mp4")
        c2.download_button("ดาวน์โหลด CSV", pd.DataFrame(result["rows"]).to_csv(index=False).encode("utf-8-sig"), "ripen_video_results.csv", "text/csv")

elif page == "ผลล่าสุด":
    st.markdown('<div class="split-title"><h2>ผลการวิเคราะห์ล่าสุด</h2><span>SESSION RESULTS</span></div><p class="sub">ผลจะอยู่ในหน้านี้ระหว่างที่เปิดเว็บรอบปัจจุบัน</p>', unsafe_allow_html=True)
    results = st.session_state.get("image_results", [])
    if not results:
        st.markdown('<div class="empty-state"><span class="empty-icon">◉</span><h3>ยังไม่มีผลการวิเคราะห์</h3><p>เริ่มจากแท็บสตูดิโอภาพ อัปโหลดรูปกล้วย แล้วผลลัพธ์จะมาอยู่ที่นี่</p></div>', unsafe_allow_html=True)
    else:
        rows = [row for item in results for row in item["rows"]]
        counts = pd.Series([row["class"] for row in rows]).value_counts() if rows else pd.Series(dtype=int)
        cols = st.columns(4)
        for i, col in enumerate(cols):
            col.metric(THAI[i], int(counts.get(NAMES[i], 0)))
        if rows:
            st.markdown('<div class="split-title"><h2>สัดส่วนที่ตรวจพบ</h2><span>DETECTIONS</span></div>', unsafe_allow_html=True)
            show_ripeness_counts(counts)
            st.download_button("ดาวน์โหลดข้อมูลทั้งหมด CSV", pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"), "ripen_results.csv", "text/csv")
        else:
            st.warning("วิเคราะห์ภาพแล้ว แต่ไม่พบกล้วยที่ผ่านเกณฑ์ ลองลดความมั่นใจขั้นต่ำในแถบตั้งค่า")
        st.markdown('<div class="split-title"><h2>ภาพที่ประมวลผล</h2><span>IMAGE GALLERY</span></div>', unsafe_allow_html=True)
        for i, item in enumerate(results):
            with st.container(border=True):
                st.markdown(f"#### {html.escape(item['name'])}")
                left, right = st.columns([2.1, 1])
                left.image(item["image"], use_container_width=True)
                right.metric("ตรวจพบ", len(item["rows"]))
                if item["rows"]:
                    right.markdown(detection_table_html(item["rows"]), unsafe_allow_html=True)
                right.download_button("ดาวน์โหลดภาพ PNG", item["image"], f"ripen_{Path(item['name']).stem}.png", "image/png", key=f"download_{i}")
        if st.button("ล้างผลในรอบนี้"):
            del st.session_state["image_results"]
            st.rerun()

else:
    st.markdown('<div class="hero"><div class="hero-copy"><div class="kicker">PROJECT / BEHIND THE MODEL</div><h1>หนึ่งภาพ.<br>สี่ระดับความสุก.</h1><p>เว็บนี้ใช้โมเดล YOLO26s ที่ฝึกตรวจระดับความสุกของกล้วย เพื่อเปลี่ยนผลการตรวจเป็นเครื่องมือที่ใช้งานได้ง่าย</p><div class="hero-detail"><span>● YOLO26s</span><span>4 RIPENESS STAGES</span></div></div>'+banana_animation()+'</div>', unsafe_allow_html=True)
    st.markdown('<div class="split-title"><h2>คลาสที่ระบบรู้จัก</h2><span>RIPENESS STAGES / 01—04</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="class-row">'+''.join(f'<div class="class-card"><span class="dot" style="background:{COLORS[i]}"></span><b>{THAI[i]}</b><small>{NAMES[i]}</small></div>' for i in range(4))+'</div>', unsafe_allow_html=True)
    st.markdown('<div class="split-title"><h2>วิธีเริ่มใช้งาน</h2><span>SETUP GUIDE</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="how"><div><span>01</span><b>ติดตั้งแพ็กเกจ</b><small>เปิด PowerShell ในโฟลเดอร์ ML และติดตั้ง requirements-app.txt</small></div><div><span>02</span><b>โมเดลพร้อมแล้ว</b><small>ไฟล์ YOLOV26/weights/best.pt รวมอยู่ในชุดนี้</small></div><div><span>03</span><b>เปิดเว็บ</b><small>รัน streamlit run app.py แล้วเลือกภาพหรือวิดีโอ</small></div></div>', unsafe_allow_html=True)
    for family in ("YOLOV26",):
        with st.expander(f"{family} · รายงานการทดลอง"):
            st.write("โมเดลที่พบ:", "YOLO26s · best.pt" if model_ready else "ยังไม่มี")
            for report in ("05_experiments.json", "06_evaluation.json"):
                path = ROOT / family / "reports" / report
                if path.is_file():
                    try:
                        with st.expander(report):
                            st.json(json.loads(path.read_text(encoding="utf-8")))
                    except (OSError, ValueError) as exc:
                        st.caption(f"อ่าน {report} ไม่ได้: {exc}")
            if not (ROOT / family / "reports").exists():
                st.caption("ยังไม่มีรายงานผลการเทรนใน ZIP")

st.markdown('<div class="footer-note">RIPEN LAB · ผลการตรวจเป็นการประเมินจากโมเดลที่ฝึกไว้ ตรวจทานผลก่อนนำไปใช้ตัดสินใจจริง</div>', unsafe_allow_html=True)
