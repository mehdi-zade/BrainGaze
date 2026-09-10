import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
import os

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(f'''
        <w:tblBorders {nsdecls("w")}>
            <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:insideV w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            <w:left w:val="none"/>
            <w:right w:val="none"/>
        </w:tblBorders>
    ''')
    tblPr.append(borders)

def make_b_nazanin_para(p, text, font_size=10.5, bold=False, color_rgb=(0,0,0), align=WD_ALIGN_PARAGRAPH.RIGHT):
    p.alignment = align
    run = p.add_run(text)
    run.font.name = 'B Nazanin'
    run.font.size = Pt(font_size)
    run.bold = bold
    run.font.color.rgb = RGBColor(*color_rgb)
    
    # Enable RTL for run and paragraph
    pPr = p._p.get_or_add_pPr()
    bidi = parse_xml(f'<w:bidi {nsdecls("w")}/>')
    pPr.append(bidi)
    
    rPr = run._r.get_or_add_rPr()
    rFonts = parse_xml(f'<w:rFonts {nsdecls("w")} w:cs="B Nazanin" w:ascii="B Nazanin" w:hAnsi="B Nazanin"/>')
    rPr.append(rFonts)

def create_persian_dataset_table_docx(output_path):
    doc = docx.Document()
    
    # Page Setup (Landscape for detailed table)
    section = doc.sections[0]
    section.orientation = docx.enum.section.WD_ORIENT.LANDSCAPE
    new_width, new_height = section.page_height, section.page_width
    section.page_width = new_width
    section.page_height = new_height
    section.top_margin = Inches(0.6)
    section.bottom_margin = Inches(0.6)
    section.left_margin = Inches(0.6)
    section.right_margin = Inches(0.6)

    # Title
    p_title = doc.add_paragraph()
    make_b_nazanin_para(p_title, "جدول (۲-۴): مقایسه تفکیکی دادگان‌های سه‌گانه پایه با دادگان ترکیبی و منطبق BGD_Dataset", font_size=14, bold=True, color_rgb=(15, 23, 42), align=WD_ALIGN_PARAGRAPH.CENTER)

    # Data
    headers = [
        "🌟 دادگان تلفیقی BGD_Dataset\n(BrainGaze-Diffusion)",
        "۳. دادگان ردیابی چشم\nSALICON [2]",
        "۲. دادگان سیگنال مغزی\nAlljoined1 [14]",
        "۱. دادگان محرک‌های بصری\nMS COCO 2014 [25]",
        "مشخصه / پارامتر فنی"
    ]

    rows_data = [
        ("نوع رسانه / مدالیته اصلی", "تصویر طبیعی رنگی RGB", "سیگنال ۳۲ کاناله EEG", "نقشه دوبعدی تراکم تثبیت نگاه", "سه‌گانه منطبق: (تصویر + EEG + نقشه نگاه)"),
        ("طراحان اولیه و سال انتشار", "Lin و همکاران (Microsoft, 2014)", "Xu و همکاران (arXiv, 2024)", "Huang و همکاران (CVPR 2015)", "عبدالله‌زاده (دانشگاه تهران، ۱۴۰۵)"),
        ("حجم کل نمونه‌های اولیه", "حدود ۱۲۳,۰۰۰ تصویر", "حدود ۱۰۰,۰۰۰ آزمایه ثبت EEG", "۱۰,۰۰۰ نقشه نگاه کالیبره‌شده", "۳۶,۲۷۵ سه‌گانه منطبق کامل"),
        ("نسبت افراز (Train / Test)", "تقسیم‌بندی استاندارد COCO", "فایل‌های خام Parquet", "تقسیم‌بندی ۱۰ هزارتایی", "۸۵٪ آموزش (۳۰,۸۳۳) / ۱۵٪ ارزیابی (۵,۴۴۲)"),
        ("تعداد ناظرین انسانی (N)", "نامشخص (کاربران Amazon MTurk)", "۲۰ ناظر انسانی (S1 ... S20)", "بیش از ۱,۳۰۰ ناظر انسانی", "۲۰ ناظر انسانی (در ۳ گونه شناختی متمایز)"),
        ("ابعاد محرک تصویری", "متغیر (معمولاً 640 x 480)", "نمایش روی نمایشگر ۶۰ هرتز", "640 x 480 پیکسل", "نرمال‌شده 224 x 224 x 3 پیکسل (ImageNet)"),
        ("پنجره زمانی و نمونه‌برداری", "تصویر ثابت دوبعدی", "۵۰۰ میلی‌ثانیه پاسخ (250Hz)", "انباشت زمانی نگاه در مشاهده آزاد", "پنجره ۵۰۰ میلی‌ثانیه‌ای (0-500ms)، ۲۵۰ نقطه زمانی"),
        ("پیکربندی کانال‌های EEG", "ندارد", "۱۲۸ / ۶۴ کانال خام", "ندارد", "۳۲ کانال استاندارد (سیستم ۱۰-۲۰ بین‌المللی)"),
        ("پیش‌پردازش و فیلتراسیون", "ندارد", "ولتاژ خام پوست سر (uV)", "هموارسازی گوسی (sigma = 1 deg)", "فیلتر باترورث ۰.۱ تا ۴۵ هرتز + مقیاس‌دهی (10^6)"),
        ("ابعاد و دامنه نقشه هدف", "ندارد", "ندارد", "640 x 480 پیوسته در [0, 1]", "نقشه احتمالی نرمال 224 x 224 x 1 (sum Y_uv = 1)"),
        ("کلید ارتباطی انطباق", "شناسه ۱۲ رقمی coco_id", "متاداده coco_id در Parquet", "متاداده image_id", "نگاشت کلید خارجی یکپارچه coco_id"),
        ("امواج زیستی زیرین (ERP)", "ندارد", "موج P100 (100ms) و P300 (300ms)", "خوشه‌بندی نقاط نگاه", "پتانسیل‌های P100 (بینایی V1)، N170 (چهره)، P300 (توجه)"),
        ("نقش در معماری BrainGaze", "ویژگی‌های پایین به بالا (Bottom-Up)", "روتر شناختی بالا به پایین (alpha)", "توزیع واقعیت زمینی (Ground Truth)", "ضرب دوخطی پارسوال: Y = sum alpha_k * M_k")
    ]

    # Create Table
    table = doc.add_table(rows=len(rows_data) + 1, cols=5)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table, color="CBD5E1", sz="4")

    # Set Column Widths (total ~ 10 inches)
    widths = [Inches(2.5), Inches(1.8), Inches(1.8), Inches(1.8), Inches(2.1)]

    # Header Row
    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].width = widths[i]
        set_cell_margins(hdr_cells[i], top=140, bottom=140, left=100, right=100)
        hdr_cells[i].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        
        # Highlight our dataset header column
        if i == 0:
            set_cell_background(hdr_cells[i], "1E1B4B") # Dark Indigo
            color = (255, 255, 255)
        else:
            set_cell_background(hdr_cells[i], "0F172A") # Slate 900
            color = (255, 255, 255)

        p = hdr_cells[i].paragraphs[0]
        make_b_nazanin_para(p, title, font_size=10, bold=True, color_rgb=color, align=WD_ALIGN_PARAGRAPH.CENTER)

    # Data Rows
    for r_idx, row in enumerate(rows_data):
        row_cells = table.rows[r_idx + 1].cells
        # Fill cells from left to right (0: BGD, 1: SALICON, 2: Alljoined1, 3: COCO, 4: Parameter Name)
        cells_values = [row[4], row[3], row[2], row[1], row[0]]
        
        bg_color = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"

        for c_idx, val in enumerate(cells_values):
            cell = row_cells[c_idx]
            cell.width = widths[c_idx]
            set_cell_margins(cell, top=100, bottom=100, left=100, right=100)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

            # Special column highlights
            if c_idx == 0: # BGD Dataset Column
                set_cell_background(cell, "F1F5F9" if r_idx % 2 == 1 else "EEF2FF")
                text_color = (30, 27, 75) # Dark Indigo Text
                is_bold = True
                align = WD_ALIGN_PARAGRAPH.RIGHT
            elif c_idx == 4: # Parameter Name Column
                set_cell_background(cell, "F1F5F9")
                text_color = (15, 23, 42) # Slate 900 Text
                is_bold = True
                align = WD_ALIGN_PARAGRAPH.RIGHT
            else:
                set_cell_background(cell, bg_color)
                text_color = (51, 65, 85) # Slate 700 Text
                is_bold = False
                align = WD_ALIGN_PARAGRAPH.RIGHT

            p = cell.paragraphs[0]
            make_b_nazanin_para(p, val, font_size=9.5, bold=is_bold, color_rgb=text_color, align=align)

    doc.save(output_path)
    print(f"Document successfully created at: {output_path}")

if __name__ == "__main__":
    out_dir = r"c:\Users\Mahdi Abdollahzadeh\Desktop\Bachelors Thesis\MS COCO\All Joined\Alljoined1\docs\thesis"
    out_file = os.path.join(out_dir, "persian_dataset_specification_table.docx")
    create_persian_dataset_table_docx(out_file)
