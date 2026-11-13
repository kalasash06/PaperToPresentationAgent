"""Turn the Slide-Generator Agent's JSON into a downloadable .pptx file."""
from io import BytesIO

from pptx import Presentation
from pptx.util import Pt


def build_pptx(deck_title: str, slides: list) -> bytes:
    prs = Presentation()
    if not slides:
        slides = [{"title": deck_title, "bullets": [], "notes": ""}]

    for i, s in enumerate(slides):
        title = str(s.get("title", "")).strip() or f"Slide {i + 1}"
        bullets = [str(b).strip() for b in s.get("bullets", []) if str(b).strip()]

        if i == 0:
            slide = prs.slides.add_slide(prs.slide_layouts[0])  # title layout
            slide.shapes.title.text = title
            slide.placeholders[1].text = "\n".join(bullets[:2])
        else:
            slide = prs.slides.add_slide(prs.slide_layouts[1])  # title + content layout
            slide.shapes.title.text = title
            slide.shapes.title.text_frame.paragraphs[0].font.size = Pt(32)
            tf = slide.placeholders[1].text_frame
            tf.clear()
            for j, bullet in enumerate(bullets):
                p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
                p.text = bullet
                p.font.size = Pt(22)

        notes = str(s.get("notes", "")).strip()
        if notes:
            slide.notes_slide.notes_text_frame.text = notes

    buf = BytesIO()
    prs.save(buf)
    return buf.getvalue()
