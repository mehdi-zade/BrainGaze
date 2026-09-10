"""
generate_persian_word_versions.py
==================================
Generates distinct versions of the Persian Thesis in Microsoft Word (.docx) format:

1. Version 1 (Pandoc Standard):
   - Direct Pandoc conversion with native OMML equations, tables, and images.
   
2. Version 2 (Persian Academic Standard - B Nazanin & B Titr):
   - Fully RTL (Right-to-Left) enabled for all paragraphs and tables.
   - Standard University of Tehran formatting (B Nazanin 13.5pt, Times New Roman 11.5pt, B Titr for headings).
   - 1.2 line spacing, standard paragraph spacing, justified alignment.
   - Centered figures and captions.
   - Clean monospace styling for code/diagram blocks.
   
3. Version 3 (Modern Academic - Vazirmatn & Executive Navy Styling):
   - Modern Persian typography (Vazirmatn + Segoe UI / Calibri).
   - Academic Navy Blue (#1B365D) accents on headings.
   - Beautifully styled tables with shaded headers and RTL column orientation.
"""

import os
import sys
import re
import shutil
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

import pypandoc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_MD = os.path.join(PROJECT_ROOT, "docs", "thesis", "thesis_final_draft.md")
OUT_DIR = os.path.join(PROJECT_ROOT, "docs", "thesis")

# Persian character regex pattern
PERSIAN_PATTERN = re.compile(r'[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF]')

def is_persian_text(text):
    """Returns True if text contains significant Persian characters."""
    if not text:
        return False
    persian_chars = len(PERSIAN_PATTERN.findall(text))
    return (persian_chars / (len(text.strip()) + 1e-5)) > 0.15

def set_p_rtl(p, align=WD_ALIGN_PARAGRAPH.RIGHT):
    """Sets paragraph to Right-to-Left (RTL) mode and adjusts alignment."""
    p.alignment = align
    pPr = p._p.get_or_add_pPr()
    
    # Remove existing bidi if any
    for existing_bidi in pPr.findall(qn('w:bidi')):
        pPr.remove(existing_bidi)
        
    bidi = OxmlElement('w:bidi')
    bidi.set(qn('w:val'), '1')
    pPr.append(bidi)

def set_p_ltr(p, align=WD_ALIGN_PARAGRAPH.LEFT):
    """Sets paragraph to Left-to-Right (LTR) mode."""
    p.alignment = align
    pPr = p._p.get_or_add_pPr()
    for existing_bidi in pPr.findall(qn('w:bidi')):
        pPr.remove(existing_bidi)

def set_run_font(run, persian_font="B Nazanin", latin_font="Times New Roman", size_pt=13.0, bold=None, italic=None, color_rgb=None):
    """Applies complex script and Latin font settings to a text run."""
    run.font.size = Pt(size_pt)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color_rgb:
        run.font.color.rgb = color_rgb

    rPr = run._r.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    rFonts.set(qn('w:ascii'), latin_font)
    rFonts.set(qn('w:hAnsi'), latin_font)
    rFonts.set(qn('w:cs'), persian_font)

    # Complex script font size in half-points (e.g. 13.5 pt -> 27)
    sz_val = str(int(size_pt * 2))
    for existing_szCs in rPr.findall(qn('w:szCs')):
        rPr.remove(existing_szCs)
    szCs = OxmlElement('w:szCs')
    szCs.set(qn('w:val'), sz_val)
    rPr.append(szCs)

    # Complex script bold
    if bold:
        for existing_bCs in rPr.findall(qn('w:bCs')):
            rPr.remove(existing_bCs)
        bCs = OxmlElement('w:bCs')
        rPr.append(bCs)

    # If run contains Persian, mark as complex script / RTL
    if is_persian_text(run.text):
        for existing_rtl in rPr.findall(qn('w:rtl')):
            rPr.remove(existing_rtl)
        rtl = OxmlElement('w:rtl')
        rtl.set(qn('w:val'), '1')
        rPr.append(rtl)

def set_table_rtl_and_borders(table, header_bg="F0F4F8", border_color="CCCCCC"):
    """Configures table for Right-to-Left viewing, cell padding, and styling."""
    tblPr = table._tbl.tblPr
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Enable RTL table layout (bidiVisual)
    for bidi in tblPr.findall(qn('w:bidiVisual')):
        tblPr.remove(bidi)
    bidiVisual = OxmlElement('w:bidiVisual')
    bidiVisual.set(qn('w:val'), '1')
    tblPr.append(bidiVisual)

    # Set subtle borders
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>\n'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="{border_color}"/>\n'
        f'  <w:left w:val="none"/>\n'
        f'  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="888888"/>\n'
        f'  <w:right w:val="none"/>\n'
        f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{border_color}"/>\n'
        f'  <w:insideV w:val="none"/>\n'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


# ═══════════════════════════════════════════════════════════════════════════════
# Generator Routines
# ═══════════════════════════════════════════════════════════════════════════════

def generate_v1_pandoc(out_path):
    """Generates Version 1: Direct Pandoc Output."""
    print("Generating Version 1: Pandoc Standard (.docx)...")
    cur_dir = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        pypandoc.convert_file(
            INPUT_MD,
            'docx',
            outputfile=out_path,
            extra_args=['--toc', '--resource-path=.']
        )
    finally:
        os.chdir(cur_dir)
    print(f"--> Version 1 ready: {out_path} ({os.path.getsize(out_path):,} bytes)")


def generate_v2_academic_bnazanin(base_docx, out_path):
    """Generates Version 2: Traditional Iranian Academic Standard (B Nazanin & B Titr)."""
    print(f"\nGenerating Version 2: Persian Academic Standard (B Nazanin / B Titr) -> {os.path.basename(out_path)}...")
    doc = docx.Document(base_docx)

    # Page Margins: Standard Iranian Thesis (Top 3cm, Bottom 2.5cm, Right 3cm, Left 2.5cm)
    for section in doc.sections:
        section.top_margin = Inches(1.18)      # 3.0 cm
        section.bottom_margin = Inches(0.98)   # 2.5 cm
        section.right_margin = Inches(1.18)    # 3.0 cm (binding margin for RTL)
        section.left_margin = Inches(0.98)     # 2.5 cm

    for p in doc.paragraphs:
        # Check if paragraph contains drawings (images)
        if p._p.xpath('.//w:drawing'):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(4)
            continue

        text = p.text.strip()
        if not text:
            continue

        style_name = p.style.name.lower()
        is_persian = is_persian_text(text)

        # Code blocks and mermaid source code
        if 'source code' in style_name or 'code' in style_name or 'verbatim' in style_name:
            set_p_ltr(p, WD_ALIGN_PARAGRAPH.LEFT)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            for run in p.runs:
                set_run_font(run, persian_font="Consolas", latin_font="Consolas", size_pt=9.5)
            continue

        # Captions for figures, tables, and diagrams
        is_caption = (
            text.startswith('شکل (') or text.startswith('شکل ') or
            text.startswith('جدول (') or text.startswith('جدول ') or
            text.startswith('نمودار (') or text.startswith('نمودار ') or
            text.startswith('تصویر (') or text.startswith('تصویر ')
        )
        if is_caption:
            set_p_rtl(p, WD_ALIGN_PARAGRAPH.CENTER)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.line_spacing = 1.15
            for run in p.runs:
                set_run_font(run, persian_font="B Nazanin", latin_font="Times New Roman", size_pt=11.5, bold=True)
            continue

        # Headings
        if 'heading 1' in style_name:
            if is_persian:
                set_p_rtl(p, WD_ALIGN_PARAGRAPH.RIGHT)
                for run in p.runs:
                    set_run_font(run, persian_font="B Titr", latin_font="Times New Roman", size_pt=17.0, bold=True, color_rgb=RGBColor(20, 30, 60))
            else:
                set_p_ltr(p, WD_ALIGN_PARAGRAPH.LEFT)
                for run in p.runs:
                    set_run_font(run, size_pt=16.0, bold=True)
            p.paragraph_format.space_before = Pt(16)
            p.paragraph_format.space_after = Pt(8)

        elif 'heading 2' in style_name:
            if is_persian:
                set_p_rtl(p, WD_ALIGN_PARAGRAPH.RIGHT)
                for run in p.runs:
                    set_run_font(run, persian_font="B Titr", latin_font="Times New Roman", size_pt=14.5, bold=True, color_rgb=RGBColor(35, 55, 95))
            else:
                set_p_ltr(p, WD_ALIGN_PARAGRAPH.LEFT)
                for run in p.runs:
                    set_run_font(run, size_pt=14.0, bold=True)
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(6)

        elif 'heading 3' in style_name or 'heading 4' in style_name or 'heading 5' in style_name:
            if is_persian:
                set_p_rtl(p, WD_ALIGN_PARAGRAPH.RIGHT)
                for run in p.runs:
                    set_run_font(run, persian_font="B Nazanin", latin_font="Times New Roman", size_pt=13.5, bold=True)
            else:
                set_p_ltr(p, WD_ALIGN_PARAGRAPH.LEFT)
                for run in p.runs:
                    set_run_font(run, size_pt=12.5, bold=True)
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(4)

        else:
            # Regular Paragraphs
            if is_persian:
                set_p_rtl(p, WD_ALIGN_PARAGRAPH.JUSTIFY)
                p.paragraph_format.line_spacing = 1.2
                p.paragraph_format.space_after = Pt(5)
                for run in p.runs:
                    set_run_font(run, persian_font="B Nazanin", latin_font="Times New Roman", size_pt=13.5)
            else:
                set_p_ltr(p, WD_ALIGN_PARAGRAPH.JUSTIFY)
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.space_after = Pt(5)
                for run in p.runs:
                    set_run_font(run, persian_font="B Nazanin", latin_font="Times New Roman", size_pt=11.5)

    # Style Tables
    for table in doc.tables:
        set_table_rtl_and_borders(table, header_bg="EAEAEA", border_color="B0B0B0")
        for row_idx, row in enumerate(table.rows):
            is_header = (row_idx == 0)
            for cell in row.cells:
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                # Header Shading
                if is_header:
                    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="E2E6EA"/>')
                    cell._tc.get_or_add_tcPr().append(shading)
                for p in cell.paragraphs:
                    p_text = p.text.strip()
                    if is_persian_text(p_text):
                        set_p_rtl(p, WD_ALIGN_PARAGRAPH.CENTER if is_header else WD_ALIGN_PARAGRAPH.RIGHT)
                        for run in p.runs:
                            set_run_font(run, persian_font="B Nazanin", latin_font="Times New Roman", size_pt=11.5, bold=is_header)
                    else:
                        set_p_ltr(p, WD_ALIGN_PARAGRAPH.CENTER)
                        for run in p.runs:
                            set_run_font(run, persian_font="B Nazanin", latin_font="Times New Roman", size_pt=10.5, bold=is_header)
                    p.paragraph_format.space_before = Pt(2)
                    p.paragraph_format.space_after = Pt(2)

    try:
        doc.save(out_path)
        print(f"--> Version 2 ready: {out_path} ({os.path.getsize(out_path):,} bytes)")
    except PermissionError:
        fallback_path = out_path.replace(".docx", "_updated.docx") if not out_path.endswith("_updated.docx") else out_path.replace("_updated.docx", "_new.docx")
        print(f"WARNING: {out_path} is currently locked (open in Microsoft Word).")
        print(f"Saving to fallback location: {fallback_path}")
        doc.save(fallback_path)
        print(f"--> Version 2 ready (fallback): {fallback_path} ({os.path.getsize(fallback_path):,} bytes)")


def generate_v3_modern_vazirmatn(base_docx, out_path):
    """Generates Version 3: Modern Academic Layout (Vazirmatn / Executive Navy Styling)."""
    print(f"\nGenerating Version 3: Modern Academic Layout (Vazirmatn / Executive Blue) -> {os.path.basename(out_path)}...")
    doc = docx.Document(base_docx)

    # Navy Accent Colors
    NAVY_PRIMARY = RGBColor(27, 54, 93)     # Deep Academic Navy
    NAVY_SECONDARY = RGBColor(45, 85, 125)  # Slate Blue
    CHARCOAL_BODY = RGBColor(30, 30, 30)

    for section in doc.sections:
        section.top_margin = Inches(1.18)      # 3.0 cm
        section.bottom_margin = Inches(0.98)   # 2.5 cm
        section.right_margin = Inches(1.18)    # 3.0 cm
        section.left_margin = Inches(0.98)     # 2.5 cm

    for p in doc.paragraphs:
        # Check if paragraph contains drawings (images)
        if p._p.xpath('.//w:drawing'):
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(4)
            continue

        text = p.text.strip()
        if not text:
            continue

        style_name = p.style.name.lower()
        is_persian = is_persian_text(text)

        # Code blocks and mermaid source code
        if 'source code' in style_name or 'code' in style_name or 'verbatim' in style_name:
            set_p_ltr(p, WD_ALIGN_PARAGRAPH.LEFT)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            for run in p.runs:
                set_run_font(run, persian_font="Consolas", latin_font="Consolas", size_pt=9.5)
            continue

        # Captions for figures, tables, and diagrams
        is_caption = (
            text.startswith('شکل (') or text.startswith('شکل ') or
            text.startswith('جدول (') or text.startswith('جدول ') or
            text.startswith('نمودار (') or text.startswith('نمودار ') or
            text.startswith('تصویر (') or text.startswith('تصویر ')
        )
        if is_caption:
            set_p_rtl(p, WD_ALIGN_PARAGRAPH.CENTER)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.line_spacing = 1.15
            for run in p.runs:
                set_run_font(run, persian_font="Vazirmatn", latin_font="Segoe UI", size_pt=11.0, bold=True, color_rgb=NAVY_SECONDARY)
            continue

        # Headings
        if 'heading 1' in style_name:
            if is_persian:
                set_p_rtl(p, WD_ALIGN_PARAGRAPH.RIGHT)
                for run in p.runs:
                    set_run_font(run, persian_font="Vazirmatn", latin_font="Segoe UI", size_pt=18.0, bold=True, color_rgb=NAVY_PRIMARY)
            else:
                set_p_ltr(p, WD_ALIGN_PARAGRAPH.LEFT)
                for run in p.runs:
                    set_run_font(run, size_pt=17.0, bold=True, color_rgb=NAVY_PRIMARY)
            p.paragraph_format.space_before = Pt(18)
            p.paragraph_format.space_after = Pt(8)

        elif 'heading 2' in style_name:
            if is_persian:
                set_p_rtl(p, WD_ALIGN_PARAGRAPH.RIGHT)
                for run in p.runs:
                    set_run_font(run, persian_font="Vazirmatn", latin_font="Segoe UI", size_pt=15.0, bold=True, color_rgb=NAVY_SECONDARY)
            else:
                set_p_ltr(p, WD_ALIGN_PARAGRAPH.LEFT)
                for run in p.runs:
                    set_run_font(run, size_pt=14.5, bold=True, color_rgb=NAVY_SECONDARY)
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)

        elif 'heading 3' in style_name or 'heading 4' in style_name or 'heading 5' in style_name:
            if is_persian:
                set_p_rtl(p, WD_ALIGN_PARAGRAPH.RIGHT)
                for run in p.runs:
                    set_run_font(run, persian_font="Vazirmatn", latin_font="Segoe UI", size_pt=13.5, bold=True, color_rgb=NAVY_SECONDARY)
            else:
                set_p_ltr(p, WD_ALIGN_PARAGRAPH.LEFT)
                for run in p.runs:
                    set_run_font(run, size_pt=13.0, bold=True)
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)

        else:
            # Body Paragraphs
            if is_persian:
                set_p_rtl(p, WD_ALIGN_PARAGRAPH.JUSTIFY)
                p.paragraph_format.line_spacing = 1.25
                p.paragraph_format.space_after = Pt(6)
                for run in p.runs:
                    set_run_font(run, persian_font="Vazirmatn", latin_font="Calibri", size_pt=12.5, color_rgb=CHARCOAL_BODY)
            else:
                set_p_ltr(p, WD_ALIGN_PARAGRAPH.JUSTIFY)
                p.paragraph_format.line_spacing = 1.18
                p.paragraph_format.space_after = Pt(6)
                for run in p.runs:
                    set_run_font(run, persian_font="Vazirmatn", latin_font="Calibri", size_pt=11.5, color_rgb=CHARCOAL_BODY)

    # Style Tables with Navy Theme
    for table in doc.tables:
        set_table_rtl_and_borders(table, header_bg="E8EEF5", border_color="C2D1E0")
        for row_idx, row in enumerate(table.rows):
            is_header = (row_idx == 0)
            for cell in row.cells:
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                # Modern Header Shading
                if is_header:
                    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="E8EEF5"/>')
                    cell._tc.get_or_add_tcPr().append(shd)
                elif row_idx % 2 == 1:
                    # Subtle zebra shading
                    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F9FBFC"/>')
                    cell._tc.get_or_add_tcPr().append(shd)

                for p in cell.paragraphs:
                    p_text = p.text.strip()
                    if is_persian_text(p_text):
                        set_p_rtl(p, WD_ALIGN_PARAGRAPH.CENTER if is_header else WD_ALIGN_PARAGRAPH.RIGHT)
                        for run in p.runs:
                            color = NAVY_PRIMARY if is_header else CHARCOAL_BODY
                            set_run_font(run, persian_font="Vazirmatn", latin_font="Calibri", size_pt=10.5, bold=is_header, color_rgb=color)
                    else:
                        set_p_ltr(p, WD_ALIGN_PARAGRAPH.CENTER)
                        for run in p.runs:
                            color = NAVY_PRIMARY if is_header else CHARCOAL_BODY
                            set_run_font(run, persian_font="Vazirmatn", latin_font="Calibri", size_pt=10.0, bold=is_header, color_rgb=color)
                    p.paragraph_format.space_before = Pt(3)
                    p.paragraph_format.space_after = Pt(3)

    try:
        doc.save(out_path)
        print(f"--> Version 3 ready: {out_path} ({os.path.getsize(out_path):,} bytes)")
    except PermissionError:
        fallback_path = out_path.replace(".docx", "_updated.docx") if not out_path.endswith("_updated.docx") else out_path.replace("_updated.docx", "_new.docx")
        print(f"WARNING: {out_path} is currently locked (open in Microsoft Word).")
        print(f"Saving to fallback location: {fallback_path}")
        doc.save(fallback_path)
        print(f"--> Version 3 ready (fallback): {fallback_path} ({os.path.getsize(fallback_path):,} bytes)")


def main():
    v1_path = os.path.join(OUT_DIR, "thesis_v1_pandoc_standard.docx")
    v2_updated_path = os.path.join(OUT_DIR, "thesis_v2_persian_academic_b_nazanin_updated.docx")
    v2_standard_path = os.path.join(OUT_DIR, "thesis_v2_persian_academic_b_nazanin.docx")
    v3_path = os.path.join(OUT_DIR, "thesis_v3_modern_vazirmatn_styled.docx")

    print("=" * 80)
    print("UPDATING PERSIAN THESIS MS WORD (.DOCX) EDITIONS")
    print("=" * 80)

    # 1. Version 1: Standard Pandoc (Intermediary base)
    generate_v1_pandoc(v1_path)

    # 2. Version 2: Specifically generate thesis_v2_persian_academic_b_nazanin_updated.docx
    generate_v2_academic_bnazanin(v1_path, v2_updated_path)

    # Also sync to standard v2 path if possible
    try:
        shutil.copyfile(v2_updated_path, v2_standard_path)
        print(f"--> Synchronized to: {v2_standard_path}")
    except PermissionError:
        print(f"Note: {v2_standard_path} is currently locked in Word; v2_updated_path has the latest content.")

    # 3. Version 3: Modern Academic (Vazirmatn & Executive Navy Styling)
    generate_v3_modern_vazirmatn(v1_path, v3_path)

    # Clean up temp v1 file
    if os.path.exists(v1_path):
        os.remove(v1_path)

    print("\n" + "=" * 80)
    print("SUCCESS: thesis_v2_persian_academic_b_nazanin_updated.docx successfully updated!")
    print("=" * 80)

if __name__ == "__main__":
    main()
