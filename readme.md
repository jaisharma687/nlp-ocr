# Table OCR Extraction System

Automatically extracts tabular data from PDFs and images into CSV and JSON files.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
brew install tesseract poppler

# 1. Convert PDFs to images
python pdf_to_image.py

# 2. Process all images and generate CSVs and JSONs
python main.py
```

## 📁 Folder Structure

```
Dataset/              → Input PDF files
Dataset_Images/       → Converted PNG images
output_csv/           → Generated CSV files
output_json/          → Generated JSON files
images/[name]/        → Debug images per processed file
```

## 🔄 Processing Pipeline

```
PDF → Image Conversion → Table Extraction → Grid Detection → OCR → CSV and JSON
```

### Step 1: PDF to Image (`PdfToImage.py`)
- Converts PDFs in `Dataset/` to PNG images
- Saves to `Dataset_Images/`

### Step 2: Table Extraction (`TableExtractor.py`)
- Locates table in image
- Corrects perspective/rotation
- Crops and pads table area

### Step 3: Grid Detection (`TableGridDetector.py`)
- Detects vertical/horizontal lines
- Identifies cell boundaries
- Maps exact row/column positions

### Step 4: OCR Extraction (`GridBasedOcrExtractor.py`)
- Extracts text from each cell
- Handles multi-line content
- Preprocesses cells for accuracy
- Generates CSV and JSON output