# 📄➡️🎥 Pdf2Video — PDF to Video Generation Pipeline

#everything mentioned below is for the legacy project 
#(i made it as a personal project the rest modifications are similar and took in from my another project echo class )

**Pdf2Video** is a Python-based project that converts PDF documents into videos by extracting pages, generating visual frames, optionally adding audio, and stitching everything into a final video output.

This repository focuses on the **media-generation side** of the pipeline and complements larger video processing systems.

---

## 🚀 What This Project Does

- Extracts pages from a PDF file
- Converts pages into image frames
- Generates audio (e.g., narration) for content
- Synchronizes frames and audio
- Stitches everything into a playable video
- Cleans up intermediate files

The end result is a **video representation of a PDF document**, suitable for demos, explainers, or content repurposing.

---

## 🧠 High-Level Pipeline

```
PDF File
  ↓
PDF Extraction
  ↓
Frame Generation
  ↓
Audio Generation
  ↓
Video Stitching
  ↓
Final Video Output
```

---

## 📂 Project Structure (Core Files)

- `pdf_extract.py`  
  Extracts text and/or page images from the PDF.

- `frame_generation.py`  
  Converts extracted PDF content into visual frames.

- `generate_audio.py`  
  Generates audio tracks (e.g., narration or voice-over).

- `generate_vedio.py`  
  Handles video generation from frames (filename typo retained).

- `stitch_vedio_and_clean.py`  
  Merges video and audio, then removes temporary files.

- `pipeline.py`  
  Orchestrates the full PDF → Video workflow.

- `test.py / tests.py`  
  Local testing and experimentation scripts.

---

## 🛠 Tech Stack

- Python
- FFmpeg
- PDF processing libraries
- Image processing utilities
- Audio generation tools (TTS or waveform-based)

---

## ▶️ How to Run (Basic)

1️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

2️⃣ Run the pipeline  
```bash
python pipeline.py
```

3️⃣ Output  
- Generated video will appear in the output directory (as defined in code)

---

## ⚠️ Notes & Assumptions

- This is an **experimental / MVP-style project**
- Paths and configs may be hardcoded
- Error handling is minimal
- Designed for local execution

---

## 🎯 Use Cases

- Converting slide-style PDFs into videos
- Generating explainer videos from documents
- Content repurposing (docs → media)
- Integration into larger video pipelines

---

## 🔗 Related Work

This project can be integrated with:
- Async video processing backends
- Media normalization pipelines
- Streaming and playback services

---

## 👨‍💻 Author

Built by **Yashmit Vaish**.

---

## 📌 Future Improvements

- Better CLI interface
- Config-driven pipeline
- Audio/video sync enhancements
- Integration with job-based processing systems
