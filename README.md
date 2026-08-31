# 📚 Picky — Pick Your Book

**Picky – Pick Your Book** is an AI-powered computer vision bookshelf assistant that helps users locate a requested book from a bookshelf image.

A user can upload a complete bookshelf photograph, enter a **book title, part of the title, or an author's name**, and the system searches the shelf, identifies the requested book, and highlights its location using a bounding box.

The project combines **object detection, Optical Character Recognition (OCR), deep-learning image classification, fuzzy text matching, and decision fusion** to create a practical end-to-end book localization system.

---

# 🎯 Problem Statement

Finding a specific book from a crowded bookshelf can be difficult and time-consuming, especially when:

- Books are tightly packed together.
- Book spines are partially visible.
- Titles are rotated vertically.
- Images are captured from a distance.
- Lighting conditions vary.
- Text is blurred or stylized.
- Multiple books have visually similar covers or spines.

A traditional image classifier can recognize a book when given a cropped image, but it cannot identify **where the book is located in the original bookshelf image**.

Similarly, OCR alone may fail when book titles are rotated, blurred, partially hidden, or printed using decorative fonts.

This project addresses this problem by building a hybrid AI system capable of both **recognizing and locating a requested book**.

---

# 🚀 Project Objective

The main objective of this project is to build an intelligent bookshelf search system that can:

- Detect individual book regions from a complete bookshelf image.
- Read visible titles and author names from book spines.
- Recognize books visually when OCR is unreliable.
- Understand user queries containing titles, partial titles, spelling variations, or author names.
- Identify the most likely requested book.
- Highlight the book directly on the original bookshelf image.
- Provide a simple and interactive web interface for users.

---

# 🧠 Project Summary

The system uses a **hybrid computer vision architecture**.

Instead of relying on a single model, multiple components work together:

1. **RF-DETR** detects candidate book regions.
2. **EasyOCR** reads title and author information from the detected book crops.
3. OCR is performed at multiple rotations to handle vertical book-spine text.
4. A strict OCR matching algorithm compares the detected text with known book metadata.
5. **EfficientNetB0** acts as a visual fallback when OCR is weak.
6. A fusion engine combines OCR and classifier evidence.
7. The user's query is resolved using fuzzy text matching.
8. The selected book's coordinates are mapped back to the original image.
9. The final bookshelf image is displayed with the requested book highlighted.

---

# 🏗️ System Architecture

```text
Bookshelf Image
      │
      ▼
RF-DETR
Detect Candidate Book Regions
      │
      ▼
EasyOCR
Read Title / Author
      │
      ▼
Strong OCR Match?
     / \
   YES  NO
    │    │
    │    ▼
    │  EfficientNetB0
    │  Visual Classification
    │    │
    └────┴─────┐
               ▼
          Fusion Engine
               │
               ▼
      Requested Book Identified
               │
               ▼
      Bounding Box Refinement
               │
               ▼
      ✅ FOUND + Highlighted Book
