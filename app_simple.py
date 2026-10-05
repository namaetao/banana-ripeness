"""เว็บตรวจความสุกกล้วยแบบพื้นฐาน ใช้โมเดล YOLO26s"""

from pathlib import Path
from io import BytesIO
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# วางไฟล์นี้ในโฟลเดอร์ ML เดียวกับโฟลเดอร์ YOLOV26
MODEL_FILE = Path(__file__).parent / "YOLOV26" / "weights" / "best.pt"
CLASS_NAMES = ["unripe", "semi_ripe", "ripe", "overripe"]
THAI_NAMES = ["ดิบ", "เริ่มสุก", "สุก", "สุกงอม"]


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_FILE))


def get_rows(result):
    """อ่านชื่อคลาสและความมั่นใจจากผลตรวจ"""
    rows = []
    for box in result.boxes:
        number = int(box.cls.item())
        rows.append({
            "ระดับความสุก": THAI_NAMES[number],
            "ความมั่นใจ": round(float(box.conf.item()), 3),
        })
    return rows


st.set_page_config(page_title="ตรวจความสุกกล้วย", page_icon="🍌")
st.title("🍌 ตรวจความสุกของกล้วย")
st.write("อัปโหลดภาพหรือวิดีโอ แล้วให้โมเดล YOLO26s ตรวจความสุก 4 ระดับ")

if not MODEL_FILE.is_file():
    st.error("ไม่พบ YOLOV26/weights/best.pt กรุณาวางไฟล์โมเดลก่อน")
    st.stop()

try:
    model = load_model()
    names = [model.names[i] for i in range(len(model.names))]
    if names != CLASS_NAMES:
        st.error(f"คลาสของโมเดลไม่ตรงกับงานนี้: {names}")
        st.stop()
except Exception as error:
    st.error(f"โหลดโมเดลไม่ได้: {error}")
    st.stop()

confidence = st.slider("ความมั่นใจขั้นต่ำ", 0.05, 0.95, 0.25, 0.05)
iou = st.slider("IoU ของกรอบซ้อน", 0.10, 0.90, 0.70, 0.05)
image_tab, video_tab = st.tabs(["ตรวจภาพ", "ตรวจวิดีโอ"])

with image_tab:
    photo = st.file_uploader("เลือกภาพกล้วย", type=["jpg", "jpeg", "png", "webp"])
    if photo and st.button("วิเคราะห์ภาพ"):
        st.session_state.pop("photo_result", None)
        try:
            image = Image.open(photo).convert("RGB")
            result = model.predict(image, conf=confidence, iou=iou, verbose=False)[0]
            # result.plot() เป็นภาพแบบ BGR จึงสลับเป็น RGB ก่อนแสดงผล
            marked = Image.fromarray(result.plot()[..., ::-1])
            output = BytesIO()
            marked.save(output, format="PNG")
            st.session_state["photo_result"] = (output.getvalue(), get_rows(result))
        except Exception as error:
            st.error(f"ตรวจภาพไม่สำเร็จ: {error}")

    if "photo_result" in st.session_state:
        image_bytes, rows = st.session_state["photo_result"]
        st.image(image_bytes, caption="ภาพหลังตรวจ", use_container_width=True)
        st.write(f"พบกล้วย {len(rows)} ตำแหน่ง")
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.download_button("ดาวน์โหลดภาพ PNG", image_bytes, "banana_result.png", "image/png")

with video_tab:
    video = st.file_uploader("เลือกวิดีโอ", type=["mp4"])
    fast_mode = st.checkbox("โหมดเร็ว (ย่อภาพวิดีโอและลดขนาดภาพที่โมเดลใช้)", value=True)
    step = st.selectbox("ตรวจทุกกี่เฟรม (1 แม่นยำสุด)", [1, 2, 5, 10], index=1)
    limit = st.number_input("จำนวนเฟรมที่ตรวจสูงสุด", min_value=10, max_value=3000, value=120, step=10)
    st.caption("ถ้ารันบน Streamlit Cloud เริ่มที่ 120 เฟรมก่อน โหมดเร็วและการข้ามเฟรมอาจทำให้พลาดจังหวะข้ามเส้น")

    if video and st.button("วิเคราะห์วิดีโอ"):
        st.session_state.pop("vertical_video_result", None)
        try:
            try:
                import lap  # ByteTrack ใช้แพ็กเกจนี้จับคู่ Track ID
            except ImportError:
                raise RuntimeError("ต้องติดตั้ง lap ก่อน: python -m pip install lap")
            with tempfile.TemporaryDirectory() as folder:
                input_file = Path(folder) / "input.mp4"
                raw_file = Path(folder) / "raw.mp4"
                output_file = Path(folder) / "output.mp4"
                input_file.write_bytes(video.getvalue())
                capture = cv2.VideoCapture(str(input_file))
                if not capture.isOpened():
                    raise ValueError("เปิดวิดีโอไม่ได้")

                fps = capture.get(cv2.CAP_PROP_FPS) or 25
                # โหลดใหม่ทุกคลิป เพื่อไม่ให้ ByteTrack จำ ID จากคลิปก่อน
                track_model = YOLO(str(MODEL_FILE))
                writer = None
                frame_no = 0
                checked = 0
                rows = []  # เก็บเฉพาะกล้วยที่ข้ามเส้น
                last_side = {}  # ID -> อยู่ซ้ายหรือขวาของเส้น
                counted_ids = set()
                left_count = 0
                right_count = 0
                progress = st.progress(0)

                try:
                    while checked < limit:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        frame_no += 1
                        if (frame_no - 1) % step != 0:
                            continue

                        if fast_mode:
                            height, width = frame.shape[:2]
                            ratio = min(1, 720 / max(height, width))
                            new_width = max(2, int(width * ratio) // 2 * 2)
                            new_height = max(2, int(height * ratio) // 2 * 2)
                            if (new_width, new_height) != (width, height):
                                frame = cv2.resize(frame, (new_width, new_height))

                        result = track_model.track(
                            frame, persist=True, tracker="bytetrack.yaml",
                            conf=confidence, iou=iou,
                            imgsz=416 if fast_mode else 640, verbose=False
                        )[0]
                        marked = result.plot()  # OpenCV ใช้ภาพแบบ BGR อยู่แล้ว
                        height, width = marked.shape[:2]
                        middle = width // 2

                        if result.boxes is not None and result.boxes.id is not None:
                            ids = result.boxes.id.int().cpu().tolist()
                            for box, track_id in zip(result.boxes, ids):
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                center_x = (x1 + x2) / 2
                                side = -1 if center_x < middle else 1
                                before = last_side.get(track_id)

                                if before is not None and before != side and track_id not in counted_ids:
                                    direction = "ขวา" if side == 1 else "ซ้าย"
                                    if side == 1:
                                        right_count += 1
                                    else:
                                        left_count += 1
                                    counted_ids.add(track_id)
                                    rows.append({
                                        "วินาที": round((frame_no - 1) / fps, 2),
                                        "Track ID": track_id,
                                        "ทิศทาง": direction,
                                        "ระดับความสุก": THAI_NAMES[int(box.cls.item())],
                                    })
                                last_side[track_id] = side

                        # วาดเส้นแนวตั้งกลางภาพและยอดนับลงบนทุกเฟรม
                        cv2.line(marked, (middle, 0), (middle, height - 1), (0, 255, 255), 2)
                        label = f"Crossed: {len(counted_ids)}  Left: {left_count}  Right: {right_count}"
                        cv2.putText(marked, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 0, 0), 4)
                        cv2.putText(marked, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 255, 255), 2)
                        if writer is None:
                            writer = cv2.VideoWriter(
                                str(raw_file), cv2.VideoWriter_fourcc(*"mp4v"),
                                max(1, fps / step), (width, height)
                            )
                            if not writer.isOpened():
                                raise ValueError("สร้างไฟล์ MP4 ไม่ได้")
                        writer.write(marked)

                        checked += 1
                        progress.progress(checked / limit)
                finally:
                    capture.release()
                    if writer is not None:
                        writer.release()

                if checked == 0:
                    raise ValueError("ไม่พบเฟรมที่ตรวจได้")
                # MP4 จาก OpenCV มักเปิดในเบราว์เซอร์ไม่ได้ จึงแปลงเป็น H.264
                progress.progress(1.0, text="กำลังบีบอัดวิดีโอ...")
                subprocess.run([
                    imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                    "-i", str(raw_file), "-an", "-c:v", "libx264",
                    "-preset", "veryfast" if fast_mode else "medium",
                    "-crf", "28" if fast_mode else "23",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(output_file),
                ], check=True, capture_output=True)
                st.session_state["vertical_video_result"] = (output_file.read_bytes(), rows, checked)
        except Exception as error:
            st.error(f"ตรวจวิดีโอไม่สำเร็จ: {error}")

    if "vertical_video_result" in st.session_state:
        video_bytes, rows, checked = st.session_state["vertical_video_result"]
        st.write(f"ตรวจ {checked} เฟรม · กล้วยข้ามเส้น {len(rows)} ลูก")
        st.write(f"ไปซ้าย {sum(row['ทิศทาง'] == 'ซ้าย' for row in rows)} · ไปขวา {sum(row['ทิศทาง'] == 'ขวา' for row in rows)}")
        st.caption("นับเมื่อจุดกึ่งกลางกรอบข้ามเส้นแนวตั้งกลางภาพ แต่ละ Track ID นับครั้งเดียว")
        st.video(video_bytes)
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            csv = pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig")
            st.download_button("ดาวน์โหลดข้อมูล CSV", csv, "banana_video.csv", "text/csv")
        st.download_button("ดาวน์โหลดวิดีโอ MP4", video_bytes, "banana_video.mp4", "video/mp4")
