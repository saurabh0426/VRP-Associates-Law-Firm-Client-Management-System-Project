import pytesseract
import spacy
from textblob import TextBlob
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import io
import os
import logging
import tempfile
from pdf2image import convert_from_bytes, convert_from_path
import traceback
import gc  # Garbage collection for memory management

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("ocr_service.log"),
                              logging.StreamHandler()])
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Increase Flask's maximum content length (default is 16MB)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Load NLP model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded spaCy model")
except Exception as e:
    logger.error(f"Error loading spaCy model: {str(e)}")
    # Fallback to smaller model if available
    try:
        nlp = spacy.load("en_core_web_md")
        logger.info("Loaded fallback spaCy model")
    except:
        logger.error("Failed to load any spaCy model")

# Set Tesseract path (Update if needed)
pytesseract.pytesseract.tesseract_cmd = r"D:\Tesseract-OCR\tesseract.exe"

# Set Poppler path (Update if needed)
POPPLER_PATH = r"D:\poppler-24.08.0\Library\bin"

# Increased OCR configuration limits
OCR_CONFIG = '--oem 1 --psm 3 -l eng'  # Default OCR config

# Global configuration settings
CONFIG = {
    'pdf_page_limit': 50,                 # Increased from 20 to 50 pages
    'max_image_dimension': 6000,          # Increased from 4000 to 6000 pixels
    'nlp_text_limit': 200000,             # Increased from 100000 to 200000 chars
    'sentiment_text_limit': 10000,        # Increased from 5000 to 10000 chars
    'correction_text_limit': 10000,       # Increased from 5000 to 10000 chars
    'max_processing_time': 300,           # 5 minutes max processing time per document
    # Higher DPI for better quality (default was 200)
    'pdf_dpi': 300,
    'timeout': 30,                        # Timeout for individual OCR operations in seconds
    # Enable parallel processing for multi-page docs
    'enable_parallel_processing': False,
}


def preprocess_image(image):
    """Enhance image for better OCR results"""
    try:
        # Convert to grayscale
        image = image.convert('L')

        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)

        # Apply slight blur to reduce noise
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))

        # Apply threshold to make text more distinct
        image = image.point(lambda x: 0 if x < 150 else 255, '1')

        return image
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        return image  # Return original image if preprocessing fails


def extract_text_from_image(image, retry=True):
    """Extract text from image with fallback options"""
    try:
        # Try with preprocessing first
        preprocessed = preprocess_image(image)
        text = pytesseract.image_to_string(preprocessed, config=OCR_CONFIG)

        # If text is too short or empty, try original image with different config
        if len(text.strip()) < 10 and retry:
            logger.info(
                "Initial OCR produced little text, trying alternate settings")
            # Try original image with different PSM settings
            text = pytesseract.image_to_string(image, config='--oem 1 --psm 6')

            if len(text.strip()) < 10:
                # Try with another PSM setting
                text = pytesseract.image_to_string(
                    image, config='--oem 1 --psm 4')

            # If still minimal text, try one more aggressive approach
            if len(text.strip()) < 10:
                # Try without preprocessing but with more sensitive settings
                text = pytesseract.image_to_string(
                    image, config='--oem 1 --psm 3 --dpi 300')

        # Free up memory
        del preprocessed
        gc.collect()

        return text
    except Exception as e:
        logger.error(f"Text extraction error: {str(e)}")
        return ""


def process_text(text):
    """Enhance OCR output with NLP: Named Entity Recognition, Sentiment, and Summarization."""
    if not text or len(text.strip()) == 0:
        return {
            "original_text": "",
            "corrected_text": "",
            "summary": "",
            "sentiment": "Neutral",
            "entities": []
        }

    try:
        # Clean text (remove some common OCR artifacts)
        text = text.replace('|', 'I').replace('\\', '').replace('~', '-')

        # Additional cleaning of common OCR errors
        text = text.replace('  ', ' ').replace(',,', ',').replace('..', '.')

        # Run NLP processing
        # Limit size for very large texts
        text_for_nlp = text[:CONFIG['nlp_text_limit']] if len(
            text) > CONFIG['nlp_text_limit'] else text
        doc = nlp(text_for_nlp)

        # Extract named entities
        entities = [{"text": ent.text, "label": ent.label_}
                    for ent in doc.ents]

        # Sentiment analysis using TextBlob
        # Analyze with increased character limit
        sentiment_text = text[:CONFIG['sentiment_text_limit']] if len(
            text) > CONFIG['sentiment_text_limit'] else text
        sentiment_score = TextBlob(sentiment_text).sentiment.polarity
        sentiment = "Positive" if sentiment_score > 0.1 else "Negative" if sentiment_score < -0.1 else "Neutral"

        # Summarization: Extract key sentences based on entity density
        sentences = [sent.text.strip() for sent in doc.sents]

        # More sophisticated summary extraction
        if len(sentences) <= 5:  # Increased from 3 to 5
            summary = " ".join(sentences)
        else:
            # Calculate sentence importance based on entities and keywords
            sentence_scores = []
            for i, sent in enumerate(doc.sents):
                score = 0
                # Count entities
                score += sum(1 for _ in sent.ents) * \
                    2  # Higher weight for entities

                # Count non-stopwords
                score += sum(1 for token in sent if not token.is_stop and token.is_alpha)

                # Bonus for position (first few sentences often important)
                if i < 3:
                    score += (3 - i)

                sentence_scores.append((sent.text, score))

            # Get top 5 sentences (increased from 3)
            top_sentences = sorted(
                sentence_scores, key=lambda x: x[1], reverse=True)[:5]

            # Sort them by original document order
            original_order = []
            for sent_text, _ in top_sentences:
                for i, sent in enumerate(sentences):
                    if sent_text == sent:
                        original_order.append((i, sent))
                        break

            original_order.sort(key=lambda x: x[0])
            summary = " ".join(text for _, text in original_order)

        # For very long texts, don't attempt spelling correction (too slow and error-prone)
        if len(text) < CONFIG['correction_text_limit']:
            try:
                corrected_text = str(TextBlob(text).correct())
            except:
                logger.warning(
                    "TextBlob correction failed, using original text")
                corrected_text = text
        else:
            corrected_text = text
            logger.info("Text too long for spelling correction")

        # Free memory
        del doc
        gc.collect()

        return {
            "original_text": text.strip(),
            "corrected_text": corrected_text.strip(),
            "summary": summary.strip(),
            "sentiment": sentiment,
            # Limit number of entities returned (for very entity-rich texts)
            "entities": entities[:500]
        }
    except Exception as e:
        logger.error(
            f"Text processing error: {str(e)}\n{traceback.format_exc()}")
        # Return basic output if NLP processing fails
        return {
            "original_text": text.strip(),
            "corrected_text": text.strip(),
            # Increased summary length
            "summary": text[:1000].strip() + "..." if len(text) > 1000 else text.strip(),
            "sentiment": "Neutral",
            "entities": []
        }


@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        file_ext = file.filename.split('.')[-1].lower()
        extracted_text = ""

        logger.info(f"Processing file: {file.filename}, type: {file_ext}")

        if file_ext in ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp', 'gif']:
            # Process image
            try:
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes))

                # Check if image is very large and resize if necessary
                max_dim = CONFIG['max_image_dimension']
                if image.width > max_dim or image.height > max_dim:
                    logger.info(
                        f"Resizing large image: {image.width}x{image.height}")
                    scale_factor = min(max_dim/image.width,
                                       max_dim/image.height)
                    new_width = int(image.width * scale_factor)
                    new_height = int(image.height * scale_factor)
                    image = image.resize(
                        (new_width, new_height), Image.LANCZOS)

                extracted_text = extract_text_from_image(image)
                logger.info(
                    f"Extracted {len(extracted_text)} characters from image")

                # Free memory
                del image
                gc.collect()

            except Exception as e:
                logger.error(
                    f"Image processing error: {str(e)}\n{traceback.format_exc()}")
                return jsonify({'error': f'Image processing error: {str(e)}'}), 500

        elif file_ext == 'pdf':
            # Convert PDF pages to images
            try:
                # Save to temp file to avoid memory issues with large PDFs
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                    temp_pdf.write(file.read())
                    temp_pdf_path = temp_pdf.name

                try:
                    # First try with temp file path - with higher DPI setting
                    images = convert_from_path(
                        temp_pdf_path,
                        poppler_path=POPPLER_PATH,
                        dpi=CONFIG['pdf_dpi'])  # Higher DPI for better quality
                except Exception as path_error:
                    logger.warning(
                        f"PDF path conversion failed: {str(path_error)}, trying bytes method")
                    # If that fails, try the bytes method
                    with open(temp_pdf_path, 'rb') as f:
                        images = convert_from_bytes(
                            f.read(),
                            poppler_path=POPPLER_PATH,
                            # Higher DPI for better quality
                            dpi=CONFIG['pdf_dpi'])

                logger.info(f"Converted PDF to {len(images)} images")

                # Limit pages for very large documents (increased limit)
                page_limit = min(CONFIG['pdf_page_limit'], len(images))
                if len(images) > page_limit:
                    logger.info(
                        f"PDF has {len(images)} pages, limiting to first {page_limit}")

                for i, img in enumerate(images[:page_limit]):
                    logger.info(f"Processing PDF page {i+1}")
                    page_text = extract_text_from_image(img)
                    extracted_text += page_text + "\n\n--- Page Break ---\n\n"

                    # Free memory after processing each page
                    del img
                    gc.collect()

                # Process remaining pages if we have many
                if len(images) > page_limit:
                    remaining_count = len(images) - page_limit
                    extracted_text += f"\n\n--- {remaining_count} more pages not processed due to page limit ---\n\n"

                # Remove temp file
                os.unlink(temp_pdf_path)

                # Free memory
                del images
                gc.collect()

                logger.info(
                    f"Extracted {len(extracted_text)} characters from PDF")
            except Exception as e:
                logger.error(
                    f"PDF processing error: {str(e)}\n{traceback.format_exc()}")
                return jsonify({'error': f'PDF processing error: {str(e)}'}), 500

        else:
            return jsonify({'error': 'Unsupported file type'}), 400

        # Check if we got any text
        if not extracted_text or len(extracted_text.strip()) < 10:
            logger.warning("Very little text extracted, returning error")
            return jsonify({
                'warning': 'Document produced minimal text. It may be empty, have poor image quality, or contain non-text elements.',
                'original_text': extracted_text.strip(),
                'corrected_text': extracted_text.strip(),
                'summary': '',
                'sentiment': 'Neutral',
                'entities': []
            })

        # Apply NLP processing to extracted text
        nlp_results = process_text(extracted_text)
        logger.info("NLP processing completed successfully")

        return jsonify(nlp_results)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'healthy', 'config': CONFIG}), 200


@app.route('/config', methods=['GET', 'POST'])
def manage_config():
    """Endpoint to get or update OCR configuration settings"""
    if request.method == 'GET':
        return jsonify(CONFIG), 200

    elif request.method == 'POST':
        try:
            new_config = request.json
            # Validate and update config
            for key, value in new_config.items():
                if key in CONFIG:
                    # Simple validation - make sure values are of same type
                    if isinstance(value, type(CONFIG[key])):
                        CONFIG[key] = value

            return jsonify({'message': 'Configuration updated', 'config': CONFIG}), 200
        except Exception as e:
            return jsonify({'error': f'Failed to update config: {str(e)}'}), 400


if __name__ == '__main__':
    logger.info("Starting OCR service with increased limits")
    app.run(debug=True, host='0.0.0.0', port=5000)
