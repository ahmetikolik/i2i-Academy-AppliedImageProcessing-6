import cv2
import numpy as np
import easyocr

def preprocess_image(image_path):
    """
    1. Görseli yükler, gri tonlamaya çevirir, bulanıklaştırır ve Canny ile kenarları bulur.
    Ödev isterlerine birebir uygundur.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(blurred, 30, 200)
    
    return img, edged

def find_license_plate_contour(edged_image):
    """
    2. Kenar (Edge) haritası üzerinden konturları bulur ve dikdörtgen şekli arar.
    """
    # findContours fonksiyonu düzeltilmiş haliyle
    contours, _ = cv2.findContours(edged_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    plate_contour = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
        
        if len(approx) == 4:
            plate_contour = approx
            break
            
    return plate_contour

def extract_and_read_plate(original_image, plate_contour):
    """
    3. Plakayı kırpar, OCR başarısını artırmak için iyileştirir ve EasyOCR ile okur.
    """
    if plate_contour is None:
        print("[UYARI] Dikdörtgen plaka çerçevesi (kontur) tespit edilemedi!")
        return None

    # Plakanın etrafındaki kutuyu al ve Orijinal görselden kırp (Crop)
    x, y, w, h = cv2.boundingRect(plate_contour)
    cropped_plate = original_image[y:y+h, x:x+w]
    
    # --- OCR İYİLEŞTİRME ADIMLARI (Daha iyi okuması için) ---
    # 1. Kırpılan plakayı 2 kat büyüt
    cropped_enlarged = cv2.resize(cropped_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 2. Gri tonlamaya çevir ve bulanıklaştır
    cropped_gray = cv2.cvtColor(cropped_enlarged, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cropped_gray, (5, 5), 0)
    
    # 3. Otsu Thresholding ile net Siyah-Beyaz'a (Binarization) çevir
    _, binary_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. EasyOCR okuyucusunu başlat
    reader = easyocr.Reader(['en'], gpu=False)
    
    # Sadece harf ve rakamlara izin ver (Noktalama işaretlerini engelle)
    results = reader.readtext(binary_img, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    if not results:
        return None
        
    # Okunan metinleri birleştir (bazen harfleri ayrı ayrı okuyabilir)
    plate_text = "".join([res[1] for res in results])
    return plate_text

def main():
    # Kendi bilgisayarınızdaki dosya yolu
    image_path = r"C:\Users\ahmet\Desktop\OCR_i2i\plaka.jpg"
    
    try:
        print("İşlem başlatıldı, resim işleniyor...")
        
        original_img, edged_img = preprocess_image(image_path)
        
        plate_contour = find_license_plate_contour(edged_img)
        
        detected_text = extract_and_read_plate(original_img, plate_contour)
        
        if detected_text:
            print(f"\n[BAŞARILI] Tespit Edilen Plaka: {detected_text.strip()}")
            cv2.imshow("Crop", cropped_plate)
        else:
            print("\n[HATA] Plaka bulunamadı veya üzerindeki metin okunamadı.")
            
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()