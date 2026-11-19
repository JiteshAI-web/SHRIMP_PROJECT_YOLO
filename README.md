# 🦐 YOLOv8 Shrimp Detection & Aquaculture Monitoring System

A complete end-to-end computer vision and aquaculture monitoring system that detects shrimps on circular feeding checktrays using a fine-tuned **YOLOv8** model, measures orientation-aware lengths and calculated biomass, stores all images and detection metrics in an indexed **PostgreSQL** database, and visualizes them on a modern, responsive web dashboard with **Ollama AI** aquaculture doctor interpretation.

---

## 🌟 Key Features

- **YOLOv8 Detection & Counting**: Detects shrimp in submerged checktrays with bounding boxes and confidence scores (`best.pt`).
- **Orientation-Aware Measurement**: Computes true shrimp length (cm) based on tray diameter calibration ($70\text{ cm}$), accurately accounting for horizontal, vertical, and diagonal shrimp postures.
- **Biomass & Weight Calculation**: Uses standard aquaculture allometric growth formula ($W = 0.0108 \times L^{2.6935}$) to estimate shrimp weight and sample biomass.
- **High-Performance PostgreSQL Storage**: Stores binary image data and rich detection metadata with B-Tree indexes, enabling `<100 ms` dashboard queries.
- **Responsive Web Dashboard**: Real vs. AI bounding-box comparison, size uniformity analytics, growth trend graphs, and a click-to-inspect fullscreen lightbox.
- **AI Aquaculture Doctor (Ollama)**: Local LLM interpretation analyzing shrimp size spread, stocking density, health risks, and actionable farmer recommendations.

---

## 📁 Project Structure

```
YOLOV8_SHRIMP/
├── requirements.txt            # Python dependencies (cross-platform)
├── .gitignore                  # Clean ignore list (no venvs or heavy checkpoints)
├── .env.example                # Database & environment template
├── run_linux.sh                # One-click launcher for Linux
├── run_windows.bat             # One-click launcher for Windows
├── README.md                   # Complete documentation
└── shrimp_detection/
    ├── check_image_counts.py   # CLI tool to check images in database
    ├── database/               # Database models & utilities
    │   ├── config.py           # DB configuration (env var support)
    │   ├── models.py           # SQLAlchemy ORM (RealImage, AIImage, indexes)
    │   ├── image_storage.py    # Binary image & measurement storage helper
    │   ├── optimize_indexes.py # PostgreSQL B-Tree index optimization script
    │   └── store_count_length_images.py # Bulk image & detection database populator
    ├── scripts/                # Detection, measurement & training scripts
    │   ├── shrimp_count_length_db_store.py # Live detection + DB storage
    │   ├── shrimp_count_length.py          # Standalone count & length
    │   ├── shrimp_detect.py                # Detection with labels
    │   ├── train.py                        # YOLOv8 training script
    │   └── runs/runs/detect/train/weights/best.pt # Trained YOLOv8 model weights
    ├── test_images/            # Test checktray images
    ├── data/                   # Train and validation datasets
    ├── outputs/
    │   └── detection_results.json # Precomputed YOLOv8 detection results
    └── webapp/                 # Flask web dashboard
        ├── app.py              # Flask server & REST API
        ├── prompts.py          # Aquaculture prompt templates
        ├── llm_client.py       # Ollama client (smart vision/metrics fallback)
        ├── test_llm_setup.py   # Ollama connection & model verification
        └── templates/
            └── index.html      # Responsive dashboard UI (lightbox, 1:1 checktrays)
```

---

## 🚀 Quick Start (Clone & Run on Any System)

### Prerequisites
- **Python 3.10+**
- **PostgreSQL** (local or remote)
- *(Optional)* **Ollama** with `llama3.1:8b` or `llava:13b` for AI interpretation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/YOLOV8_SHRIMP.git
cd YOLOV8_SHRIMP
```

### 2. Configure Database
Copy the example environment file:
```bash
cp .env.example .env
```
Edit `.env` (or set environment variables) to match your PostgreSQL instance:
```ini
DB_HOST=192.168.1.199
DB_PORT=5432
DB_NAME=shrimp_db
DB_USER=bariflo
DB_PASSWORD=Bariflo@2401
```

### 3. Run with One-Click Launcher

**On Linux / macOS:**
```bash
chmod +x run_linux.sh
./run_linux.sh
```

**On Windows:**
Double-click `run_windows.bat` or run:
```bat
run_windows.bat
```

The script will automatically create a virtual environment, install dependencies, and start the web dashboard at **http://127.0.0.1:5000**.

---

## 🛠️ Manual Setup

If you prefer installing dependencies manually:

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. (First time only) Populate database tables and images
python shrimp_detection/database/optimize_indexes.py
python shrimp_detection/database/store_count_length_images.py --count-length

# 4. Start Flask web dashboard
cd shrimp_detection/webapp
python app.py
```

Open your browser to:
👉 **http://127.0.0.1:5000**

---

## 📊 Database Schema

The system uses two relational tables with B-Tree indexes:

### `real_images`
- `id` (Primary Key)
- `image_name`, `image_type` (`test`, `train`, `validation`)
- `image_data` (`LargeBinary` / bytea)
- `width`, `height`, `file_format`, `file_size_bytes`
- Indexes: `ix_real_images_image_type`, `ix_real_images_created_at`

### `ai_images`
- `id` (Primary Key)
- `real_image_id` (Foreign Key -> `real_images.id` on delete cascade)
- `image_data` (Annotated image with green bounding boxes & length labels)
- `shrimp_count` (Integer)
- `avg_shrimp_length_cm` (Float, orientation-aware)
- `avg_shrimp_weight_g` (Float, allometric weight)
- `measurements_json` (JSON list with individual shrimp IDs, lengths, confidence, bounding boxes)
- `pond_id`, `cycle_no`, `device_id`, `water_depth_m` (Metadata)
- Indexes: `ix_ai_images_real_image_id`, `ix_ai_images_type_created_at`, `ix_ai_images_count_created_at`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Responsive Monitoring Dashboard |
| `GET` | `/api/images` | Fetches 2 paired Real & AI checktray images + chart data (<100ms) |
| `GET` | `/api/stats` | Database summary counts grouped by image type |
| `GET` | `/api/latest-detection` | Latest detection count, average length, model, and timestamp |
| `GET` | `/api/image/<id>?type=real\|ai` | Direct cached image stream |
| `GET` | `/api/health` | System health, DB latency, and Ollama status |
| `POST` | `/api/generate-interpretation` | AI aquaculture doctor analysis via Ollama |

---

## 👨‍⚕️ Ollama AI Integration

To use the **"Generate Interpretation"** button:
1. Install [Ollama](https://ollama.ai).
2. Pull a model:
   ```bash
   ollama pull llama3.1:8b   # Metrics & aquaculture expert mode
   # or
   ollama pull llava:13b     # Vision + metrics multimodal mode
   ```
3. Run `ollama serve`. The web app automatically detects the installed model.

---

## 📄 License
This project is licensed under the MIT License.
