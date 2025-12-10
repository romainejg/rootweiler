import os
from typing import List, Tuple, Optional, Union
from io import BytesIO

from PIL import Image
import fitz  # PyMuPDF
from docx import Document
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
import openpyxl

# Type alias: (PIL.Image, suggested_filename)
ImageInfo = Tuple[Image.Image, str]


def process_pdf(file: Union[str, bytes]) -> List[ImageInfo]:
    """
    Extract images from a PDF file.

    Args:
        file: path to a PDF file (str) or PDF bytes.

    Returns:
        List of (PIL.Image, suggested_filename)
    """
    images: List[ImageInfo] = []

    if isinstance(file, (bytes, bytearray)):
        doc = fitz.open(stream=file, filetype="pdf")
    else:
        doc = fitz.open(file)

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))

            # Filter out small images
            if image.width >= 30 and image.height >= 30:
                name = f"pdf_page_{page_index + 1}_image_{img_index + 1}.png"
                images.append((image, name))

    return images


def process_word(file: Union[str, bytes]) -> List[ImageInfo]:
    """
    Extract images from a Word (.docx) file.

    Args:
        file: path to a DOCX file (str) or DOCX bytes.

    Returns:
        List of (PIL.Image, suggested_filename)
    """
    images: List[ImageInfo] = []

    if isinstance(file, (bytes, bytearray)):
        doc = Document(BytesIO(file))
    else:
        doc = Document(file)

    image_index = 0
    for rel in doc.part.rels:
        target_ref = doc.part.rels[rel].target_ref
        if "image" in target_ref:
            img_data = doc.part.rels[rel].target_part.blob
            image = Image.open(BytesIO(img_data))
            if image.width >= 30 and image.height >= 30:
                image_index += 1
                name = f"word_image_{image_index}.png"
                images.append((image, name))

    return images


def process_pptx(file: Union[str, bytes]) -> List[ImageInfo]:
    """
    Extract images from a PowerPoint (.pptx) file.

    Args:
        file: path to a PPTX file (str) or PPTX bytes.

    Returns:
        List of (PIL.Image, suggested_filename)
    """
    images: List[ImageInfo] = []

    if isinstance(file, (bytes, bytearray)):
        prs = Presentation(BytesIO(file))
    else:
        prs = Presentation(file)

    image_index = 0
    for slide_idx, slide in enumerate(prs.slides):
        for shape in slide.shapes:
            if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
                img = shape.image
                img_data = img.blob
                image = Image.open(BytesIO(img_data))
                if image.width >= 30 and image.height >= 30:
                    image_index += 1
                    name = f"pptx_slide_{slide_idx + 1}_image_{image_index}.png"
                    images.append((image, name))

    return images


def process_xlsx(file: Union[str, bytes]) -> List[ImageInfo]:
    """
    Extract images from an Excel (.xlsx) file.

    Args:
        file: path to an XLSX file (str) or XLSX bytes.

    Returns:
        List of (PIL.Image, suggested_filename)
    """
    images: List[ImageInfo] = []

    if isinstance(file, (bytes, bytearray)):
        wb = openpyxl.load_workbook(BytesIO(file))
    else:
        wb = openpyxl.load_workbook(file)

    for sheet in wb:
        image_index = 0
        # Note: using _images (private) as in your original code
        for img_obj in getattr(sheet, "_images", []):
            # openpyxl stores the image as .ref or ._data depending on version.
            # Your original code used image.ref.blob, so we keep that:
            img_data = getattr(getattr(img_obj, "ref", None), "blob", None)
            if img_data is None:
                # fallback: some versions use _data
                img_data = getattr(img_obj, "_data", None)

            if img_data is None:
                continue

            image = Image.open(BytesIO(img_data))
            if image.width >= 30 and image.height >= 30:
                image_index += 1
                safe_sheet_name = sheet.title.replace(" ", "_")
                name = f"xlsx_image_{safe_sheet_name}_{image_index}.png"
                images.append((image, name))

    return images


def extract_images_from_file(file_path: str) -> List[ImageInfo]:
    """
    Convenience function: given a file path (pdf/docx/pptx/xlsx),
    detect the extension and extract all valid images.

    Args:
        file_path: path to the input file on disk.

    Returns:
        List of (PIL.Image, suggested_filename)
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        return process_pdf(file_path)
    elif ext == ".docx":
        return process_word(file_path)
    elif ext == ".pptx":
        return process_pptx(file_path)
    elif ext == ".xlsx":
        return process_xlsx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def extract_images_from_bytes(data: bytes, extension: str) -> List[ImageInfo]:
    """
    Convenience function: given file bytes and an extension, extract images.

    Useful for Streamlit uploads.

    Args:
        data: raw file bytes (from upload)
        extension: file extension including dot, e.g. '.pdf', '.docx', '.pptx', '.xlsx'

    Returns:
        List of (PIL.Image, suggested_filename)
    """
    ext = extension.lower()
    if ext == ".pdf":
        return process_pdf(data)
    elif ext == ".docx":
        return process_word(data)
    elif ext == ".pptx":
        return process_pptx(data)
    elif ext == ".xlsx":
        return process_xlsx(data)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def save_images_to_folder(images: List[ImageInfo], folder_path: str) -> None:
    """
    Save a list of (image, name) pairs to a folder.

    Args:
        images: list of (PIL.Image, filename)
        folder_path: where to save the images
    """
    os.makedirs(folder_path, exist_ok=True)

    for image, name in images:
        save_path = os.path.join(folder_path, name)
        image.save(save_path)
