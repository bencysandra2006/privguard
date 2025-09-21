import sys
import os
import re
import io
import pytesseract
import fitz  # PyMuPDF
import spacy
import cv2
from PIL import Image, ExifTags

TESSERACT_CONFIG = r"--oem 3 --psm 6"

# spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    print("[!] spaCy model not found. Run: python -m spacy download en_core_web_sm")
    raise e

# -----------------------
# Utility / helpers
# -----------------------
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF pages; OCR-render pages that have no selectable text."""
    text_parts = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[!] Could not open PDF {pdf_path}: {e}")
        return ""

    for i, page in enumerate(doc):
        try:
            page_text = page.get_text("text").strip()
        except Exception:
            page_text = ""
        if page_text:
            text_parts.append(page_text)
        else:
            # OCR fallback
            try:
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                ocr_text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
            except Exception as e:
                print(f"[!] OCR error on PDF page {i}: {e}")
                ocr_text = ""
            text_parts.append(ocr_text)
    doc.close()
    return "\n\n".join(text_parts)


def detect_pii_entities(text):
    """
    Detect PII using spaCy entities + regex (emails, phones, dates, aadhaar-like).
    Returns a list of (entity_text, label).
    """
    entities = []
    if not text:
        return entities

    # spaCy NER
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in {"PERSON", "GPE", "ORG", "DATE", "LOC"}:
            et = ent.text.strip()
            if et:
                entities.append((et, ent.label_))

    # regex patterns
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}"
    phone_pattern = r"\b\d{10}\b"
    # common date patterns: 01/01/2016 or 01-01-2016 or 1/1/16 etc.
    date_pattern = r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"
    aadhaar_pattern = r"\b\d{4}\s?\d{4}\s?\d{4}\b"

    for m in set(re.findall(email_pattern, text)):
        entities.append((m, "EMAIL"))
    for m in set(re.findall(phone_pattern, text)):
        entities.append((m, "PHONE"))
    for m in set(re.findall(date_pattern, text)):
        entities.append((m, "DATE"))
    for m in set(re.findall(aadhaar_pattern, text)):
        entities.append((m, "AADHAAR"))

    # deduplicate while preserving labels (simple)
    seen = set()
    uniq = []
    for t, l in entities:
        key = (t.strip(), l)
        if key not in seen:
            seen.add(key)
            uniq.append((t.strip(), l))
    return uniq


def redact_text_replacements(text, entities):
    """Return a redacted copy of text where occurrences of entities are replaced with [REDACTED]."""
    redacted = text
    for ent_text, _ in entities:
        if not ent_text:
            continue
        # replace all occurrences (naive; fine for prototype)
        redacted = redacted.replace(ent_text, "[REDACTED]")
    return redacted


def create_redacted_pdf(original_pdf_path, pii_entities, output_path):
    """
    Add redact annotations (black bars) over found PII strings in a PDF, apply and save.
    Best-effort: uses text search for each entity string in each page.
    """
    try:
        doc = fitz.open(original_pdf_path)
    except Exception as e:
        print(f"[!] Could not open PDF for redaction: {e}")
        return False

    # remove metadata (best-effort)
    try:
        doc.set_metadata({})
    except Exception:
        pass

    unique_entities = list({(t, l) for (t, l) in pii_entities})

    for page in doc:
        for ent_text, _ in unique_entities:
            # skip empty or extremely short
            if not ent_text or len(ent_text.strip()) < 2:
                continue
            try:
                rects = page.search_for(ent_text)
            except Exception:
                rects = []
            for r in rects:
                try:
                    page.add_redact_annot(r, fill=(0, 0, 0))
                except Exception:
                    pass
        try:
            page.apply_redactions()
        except Exception:
            pass

    try:
        doc.save(output_path)
        doc.close()
        print(f"[+] Saved redacted PDF → {output_path}")
        return True
    except Exception as e:
        print(f"[!] Failed to save redacted PDF: {e}")
        try:
            doc.close()
        except Exception:
            pass
        return False


def preprocess_for_ocr_image(img):
    """
    Improve image for OCR: convert to grayscale, increase contrast / threshold.
    Returns a single-channel (grayscale) image suitable for pytesseract.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # adaptive threshold or simple threshold; use OTSU for variable lighting
    try:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    except Exception:
        return gray


def redact_image_with_text_and_faces(image_path, output_path):
    """
    OCR the image, detect PII, black-out words that match (substring, case-insensitive),
    then blur faces. Returns (pii_entities_list, face_count).
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Could not read image: {image_path}")
        return [], 0

    # strip EXIF by re-saving (PIL)
    try:
        pil = Image.open(image_path)
        clean = Image.new(pil.mode, pil.size)
        clean.putdata(list(pil.getdata()))
        temp_no_exif = "__tmp_no_exif.png"
        clean.save(temp_no_exif)
        img = cv2.imread(temp_no_exif)
        try:
            os.remove(temp_no_exif)
        except Exception:
            pass
    except Exception:
        # if EXIF removal fails, continue with original img
        pass

    # prepare for OCR
    proc = preprocess_for_ocr_image(img)

    # Tesseract OCR: get word-level boxes
    try:
        ocr_data = pytesseract.image_to_data(proc, output_type=pytesseract.Output.DICT, config=TESSERACT_CONFIG)
    except Exception as e:
        print(f"[!] Tesseract OCR failed on image: {e}")
        return [], 0

    # combine words into full_text for PII detection
    words = [w for w in ocr_data.get("text", []) if w and w.strip()]
    full_text = " ".join(words)

    # detect entities from OCR text
    pii_entities = detect_pii_entities(full_text)

    # For each OCR word, if it matches (substring, case-insensitive) any entity -> redact that word bbox
    n_words = len(ocr_data.get("text", []))
    redacted_boxes = []
    for i in range(n_words):
        word = (ocr_data["text"][i] or "").strip()
        if not word:
            continue
        lx = ocr_data["left"][i]
        ly = ocr_data["top"][i]
        w = ocr_data["width"][i]
        h = ocr_data["height"][i]
        for ent_text, _ in pii_entities:
            # case-insensitive substring check
            try:
                if word.lower() in ent_text.lower():
                    redacted_boxes.append((lx, ly, w, h))
                    break
            except Exception:
                continue

    # Draw black rectangles over text PII
    for (x, y, w, h) in redacted_boxes:
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + w), int(y + h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)

    # Face detection & blur
    gray_for_faces = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray_for_faces, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        sub = img[y:y+h, x:x+w]
        blurred = cv2.GaussianBlur(sub, (99, 99), 30)
        img[y:y+h, x:x+w] = blurred

    # Save redacted image
    try:
        cv2.imwrite(output_path, img)
    except Exception as e:
        print(f"[!] Could not save redacted image: {e}")

    return pii_entities, len(faces)


def generate_report_per_file(report_entries, output_path="report.txt"):
    """
    report_entries: list of dicts: { 'file': filename, 'pii': [(text,label),...], 'faces': int }
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=== PRIVGUARD AUDIT REPORT ===\n\n")
            for entry in report_entries:
                f.write(f"File: {entry['file']}\n")
                f.write(f"  Faces redacted: {entry.get('faces', 0)}\n")
                f.write(f"  Text PII found: {len(entry.get('pii', []))}\n")
                if entry.get('pii'):
                    for (t, l) in entry['pii']:
                        f.write(f"    - {l}: {t}\n")
                f.write("\n")
        print(f"[+] Audit report written to {output_path}")
    except Exception as e:
        print(f"[!] Failed to write report: {e}")


# -----------------------
# MAIN
# -----------------------
def main(argv):
    if len(argv) < 2:
        print("Usage: python privguard.py <file1> <file2> ...")
        return 1

    report_entries = []

    for filepath in argv[1:]:
        if not os.path.exists(filepath):
            print(f"[!] File not found: {filepath}")
            continue

        basename = os.path.basename(filepath)
        name_noext = os.path.splitext(basename)[0]

        if filepath.lower().endswith(".pdf"):
            print(f"[*] Processing PDF: {filepath}")
            text = extract_text_from_pdf(filepath)
            pii = detect_pii_entities(text)
            redacted_text = redact_text_replacements(text, pii)

            # save redacted text
            txt_out = f"redacted_{name_noext}.txt"
            try:
                with open(txt_out, "w", encoding="utf-8") as f:
                    f.write(redacted_text)
                print(f"[+] Saved redacted text -> {txt_out}")
            except Exception as e:
                print(f"[!] Could not save redacted text: {e}")

            # create redacted pdf with black bars
            pdf_out = f"redacted_{name_noext}.pdf"
            success = create_redacted_pdf(filepath, pii, pdf_out)

            report_entries.append({'file': basename, 'pii': pii, 'faces': 0})

        elif filepath.lower().endswith((".jpg", ".jpeg", ".png")):
            print(f"[*] Processing Image: {filepath}")
            img_out = f"redacted_{name_noext}{os.path.splitext(filepath)[1]}"
            pii_list, face_count = redact_image_with_text_and_faces(filepath, img_out)
            # pii_list is list of (text,label)
            report_entries.append({'file': basename, 'pii': pii_list, 'faces': face_count})
            print(f"[+] Saved redacted image -> {img_out} (faces blurred: {face_count})")
        else:
            print(f"[!] Unsupported file type: {filepath}")
            continue

    # write combined report
    if report_entries:
        generate_report_per_file(report_entries, output_path="report.txt")
    else:
        print("[*] No files processed — nothing to report.")

    print("✅ Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))




