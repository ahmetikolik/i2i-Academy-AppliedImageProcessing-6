# License Plate Recognition (OCR) 🚗

A straightforward Python script for detecting and reading vehicle license plates from images. This project leverages **OpenCV** for image preprocessing and contour detection, and **EasyOCR** for accurate text extraction.

## Features
*   **Image Preprocessing:** Applies grayscale conversion, bilateral filtering, and Canny edge detection to highlight structural boundaries.
*   **Contour Detection:** Scans the edge map to locate the rectangular shape of the license plate.
*   **OCR Enhancement:** Crops the identified region, resizes it, and applies Otsu's thresholding to maximize EasyOCR's read accuracy.
*   **Text Extraction:** Utilizes EasyOCR with a strict alphanumeric allowlist to filter out unwanted punctuation and noise.

## Prerequisites
Make sure you have Python installed. You will need the following libraries to run the script:

```bash
pip install opencv-python numpy easyocr