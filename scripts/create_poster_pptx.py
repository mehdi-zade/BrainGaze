"""
scripts/create_poster_pptx.py
=============================
Generates the official University of Tehran Bachelor Thesis Poster in PowerPoint (.pptx) format.
Dimensions: 23.4 x 33.1 inches (ISO A1 / Academic Poster Standard)
Matches the exact template layout, cards, color palette, figures, and typography.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
ASSETS_DIR = os.path.join(FIGURES_DIR, "template_assets")
OUTPUT_PPTX = os.path.join(PROJECT_ROOT, "outputs", "poster_presentation.pptx")

# -------------------------------------------------------------------------
# Color Palette (University of Tehran & High-End Academic Theme)
# -------------------------------------------------------------------------
COLOR_PAGE_BG       = RGBColor(0xFA, 0xFA, 0xFC)  # Crisp off-white / light slate
COLOR_PRIMARY       = RGBColor(0x1A, 0x23, 0x7E)  # Deep UT Navy Blue
COLOR_ACCENT        = RGBColor(0x4A, 0x14, 0x8C)  # Deep Academic Purple
COLOR_CARD_BORDER   = RGBColor(0x4A, 0x14, 0x8C)  # Template border color
COLOR_CARD_BG       = RGBColor(0xFF, 0xFF, 0xFF)  # Pure White card fill
COLOR_HEADER_TEXT   = RGBColor(0x0F, 0x17, 0x2A)  # Slate 900
COLOR_BODY_TEXT     = RGBColor(0x1E, 0x29, 0x3B)  # Slate 800
COLOR_MUTED_TEXT    = RGBColor(0x47, 0x55, 0x69)  # Slate 600
COLOR_TABLE_HEADER  = RGBColor(0x1E, 0x29, 0x3B)  # Dark Table Header
COLOR_TABLE_ROW_ALT = RGBColor(0xF8, 0xFA, 0xFC)  # Zebra row fill
COLOR_PASSED_GREEN  = RGBColor(0x15, 0x80, 0x3D)  # Green for passed status

def build_poster():
    print("Initializing PowerPoint Presentation...")
    prs = Presentation()
    
    # 23.4 x 33.1 inches (matches template ratio 1 : 1.414)
    prs.slide_width = Inches(23.4)
    prs.slide_height = Inches(33.1)
    
    # Blank slide layout
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    
    # Background fill
    bg_shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(23.4), Inches(33.1)
    )
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = COLOR_PAGE_BG
    bg_shape.line.fill.background()

    # -------------------------------------------------------------------------
    # 1. Header Banner & Logos
    # -------------------------------------------------------------------------
    header_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.6), Inches(21.8), Inches(3.8)
    )
    header_box.fill.solid()
    header_box.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    header_box.line.color.rgb = COLOR_CARD_BORDER
    header_box.line.width = Pt(2.5)

    # University of Tehran Logo (Top Right)
    logo_ut_path = os.path.join(ASSETS_DIR, "img_1_Image19.png")
    if os.path.exists(logo_ut_path):
        slide.shapes.add_picture(logo_ut_path, Inches(19.9), Inches(0.85), width=Inches(2.3))

    # ECE / Faculty Logo (Top Left)
    logo_ece_path = os.path.join(ASSETS_DIR, "img_2_Image21.png")
    if os.path.exists(logo_ece_path):
        slide.shapes.add_picture(logo_ece_path, Inches(1.1), Inches(0.85), width=Inches(2.4))

    # Header Text (Center)
    tx_box = slide.shapes.add_textbox(Inches(3.7), Inches(0.7), Inches(16.0), Inches(3.6))
    tf = tx_box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p0 = tf.paragraphs[0]
    p0.text = "عنوان پروژه: تلفیق چندحالته ویژگی‌های ردیابی شناختی مغز و محرک‌های دیداری جهت پیش‌بینی توجه بصری (BrainGaze)"
    p0.font.bold = True
    p0.font.size = Pt(26)
    p0.font.color.rgb = COLOR_PRIMARY
    p0.alignment = PP_ALIGN.CENTER

    p1 = tf.add_paragraph()
    p1.text = "ممیزی تشخیصی و مهار قطعی پدیده «یادگیری میان‌بُر» با مسیریابی پیمانه‌ای شناختی-دیداری (CVMR)"
    p1.font.bold = True
    p1.font.size = Pt(17)
    p1.font.color.rgb = COLOR_ACCENT
    p1.alignment = PP_ALIGN.CENTER

    p2 = tf.add_paragraph()
    p2.text = "دانشجو: مهدی عبدالله‌زاده               استاد راهنما: دکتر محمدعلی اخائی"
    p2.font.bold = True
    p2.font.size = Pt(20)
    p2.font.color.rgb = COLOR_HEADER_TEXT
    p2.alignment = PP_ALIGN.CENTER

    p3 = tf.add_paragraph()
    p3.text = "دانشکده مهندسی برق و کامپیوتر، دانشگاه تهران"
    p3.font.size = Pt(17)
    p3.font.color.rgb = COLOR_MUTED_TEXT
    p3.alignment = PP_ALIGN.CENTER

    # Helper function to create section cards
    def create_card(left, top, width, height, title):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = COLOR_CARD_BG
        card.line.color.rgb = COLOR_CARD_BORDER
        card.line.width = Pt(2.0)

        tbox = slide.shapes.add_textbox(left + Inches(0.3), top + Inches(0.15), width - Inches(0.6), Inches(0.7))
        ttf = tbox.text_frame
        ttf.word_wrap = True
        ttf.margin_left = ttf.margin_top = ttf.margin_right = ttf.margin_bottom = 0
        tp = ttf.paragraphs[0]
        tp.text = title
        tp.font.bold = True
        tp.font.size = Pt(22)
        tp.font.color.rgb = COLOR_CARD_BORDER
        tp.alignment = PP_ALIGN.CENTER
        return card

    col_w = Inches(10.6)
    r_col_x = Inches(12.0)
    l_col_x = Inches(0.8)

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: Section 1 - مقدمه / خلاصه (Intro / Summary)
    # -------------------------------------------------------------------------
    create_card(r_col_x, Inches(4.7), col_w, Inches(6.0), "مقدمه / خلاصه")
    s1_box = slide.shapes.add_textbox(r_col_x + Inches(0.4), Inches(5.5), col_w - Inches(0.8), Inches(5.0))
    s1_tf = s1_box.text_frame
    s1_tf.word_wrap = True
    s1_tf.margin_left = s1_tf.margin_top = s1_tf.margin_right = s1_tf.margin_bottom = 0

    intro_points = [
        "• نوع پروژه: تحقیقاتی کاربردی و شبیه‌سازی عمیق چندوجهی (Deep Multimodal Learning) با واسط‌های مغز و رایانه (BCI).",
        "• اهداف پروژه: پیش‌بینی و بازسازی نقشه توجه بصری شخصی‌سازی‌شده ناظر با تلفیق سیگنال‌های ۳۲ کاناله EEG و محرک‌های دیداری، و مهار پدیده «فروپاشی مُدالیته» (Modality Collapse).",
        "• سوال اصلی و فرضیه: چرا مدل‌های عصبی-دیداری مرجع، سیگنال مغزی را نادیده گرفته و به میان‌بُر تصویری پناه می‌برند؟ آیا می‌توان با طراحی معماری بدون زیرفضای پوچ، استفاده از سیگنال مغز را از نظر ریاضی گریزناپذیر کرد؟",
        "• روش پاسخ: تدوین «استاندارد تشخیصی عصبی-بینایی» (NVDS) برای ممیزی رخنه‌گاه‌های FiLM و گیت‌های پسماند، و طراحی معماری نوین BrainGaze بر پایه مسیریابی پیمانه‌ای شناختی-دیداری (CVMR).",
        "• دستاوردها: مهار قطعی میان‌بُر، ارتقای همبستگی به CC = 0.8609 و دستیابی به حساسیت زیستی فعال +24.65% در برابر نویز مغزی."
    ]
    for i, pt in enumerate(intro_points):
        p = s1_tf.paragraphs[0] if i == 0 else s1_tf.add_paragraph()
        p.text = pt
        p.font.size = Pt(13.5)
        p.font.color.rgb = COLOR_BODY_TEXT
        p.alignment = PP_ALIGN.RIGHT

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: Section 2 - روش/ساختار/مدل پیشنهادی (Proposed Method)
    # -------------------------------------------------------------------------
    create_card(r_col_x, Inches(10.9), col_w, Inches(21.4), "روش/ساختار/مدل پیشنهادی")
    s2_box = slide.shapes.add_textbox(r_col_x + Inches(0.4), Inches(11.7), col_w - Inches(0.8), Inches(5.6))
    s2_tf = s2_box.text_frame
    s2_tf.word_wrap = True
    s2_tf.margin_left = s2_tf.margin_top = s2_tf.margin_right = s2_tf.margin_bottom = 0

    method_points = [
        "معماری تحول‌آفرین BrainGaze (CVMR) بر اصل «تقسیم کار دوخطی» و حذف ساختاری زیرفضای پوچ استوار است:",
        "۱. شاخه دیداری: استخراج ویژگی توسط ResNet-18 منجمد و تولید K=8 نقشه پایه مکانی متعامد (M_1 ... M_8) تحت قید تعامدی گرام و قضیه پارسوال (L_ortho).",
        "۲. شاخه شناختی مغز: عبور سیگنال ۳۲ کاناله EEG از انکودر زمانی (Conv1D) و ترانسفورمر مکانی-زمانی به همراه بردار هویت سوژه، جهت تعیین وزن‌های مسیریابی α ∈ Δ⁸ با حداکثر آنتروپی (L_div).",
        "۳. تلفیق دوخطی پیمانه‌ای: بازسازی نقشه توجه Ŷ = ∑ α_k M_k؛ طبق قضیه تعبیه هم‌اندازه پارسوال (||ΔŶ||_L² ≡ ||Δα||₂)، هر تغییر در حالت مغز الزاماً نقشه توجه را جابجا کرده و دور زدن مغز از نظر ریاضی ناممکن است.",
        "• محیط و دادگان: پیاده‌سازی در PyTorch بر روی دادگان BGD (شامل ۳,۲۳۴ نمونه ثبت همزمان EEG ناظران و تصاویر MS COCO). ارزیابی با معیارهای CC, KLD, SIM, NSS و آزمون‌های پنج‌گانه ممیزی تشخیصی NVDS."
    ]
    for i, pt in enumerate(method_points):
        p = s2_tf.paragraphs[0] if i == 0 else s2_tf.add_paragraph()
        p.text = pt
        p.font.size = Pt(13.2)
        p.font.color.rgb = COLOR_BODY_TEXT
        p.alignment = PP_ALIGN.RIGHT

    # Architecture Diagram in Section 2
    arch_img_path = os.path.join(FIGURES_DIR, "poster_architecture_diagram.png")
    if os.path.exists(arch_img_path):
        slide.shapes.add_picture(arch_img_path, r_col_x + Inches(0.4), Inches(17.6), width=col_w - Inches(0.8))

    # Mathematical Proof Box in Section 2 (below diagram)
    math_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, r_col_x + Inches(0.4), Inches(24.2), col_w - Inches(0.8), Inches(7.6))
    math_box.fill.solid()
    math_box.fill.fore_color.rgb = RGBColor(0xF8, 0xFA, 0xFC)
    math_box.line.color.rgb = RGBColor(0xCB, 0xD5, 0xE1)
    math_box.line.width = Pt(1.5)

    mb_text = slide.shapes.add_textbox(r_col_x + Inches(0.6), Inches(24.4), col_w - Inches(1.2), Inches(7.2))
    mb_tf = mb_text.text_frame
    mb_tf.word_wrap = True
    mb_tf.margin_left = mb_tf.margin_top = mb_tf.margin_right = mb_tf.margin_bottom = 0

    p_mb_title = mb_tf.paragraphs[0]
    p_mb_title.text = "★ مبانی ریاضی و حذف رخنه‌گاه زیرفضای پوچ دکودر:"
    p_mb_title.font.bold = True
    p_mb_title.font.size = Pt(14)
    p_mb_title.font.color.rgb = COLOR_PRIMARY
    p_mb_title.alignment = PP_ALIGN.RIGHT

    math_bullets = [
        "• تضمین هم‌اندازگی پارسوال: ||Ŷ₁ - Ŷ₂||_L² ≡ ||α₁ - α₂||₂ (تطابق کامل متریک اقلیدسی مغز و فضای بازسازی دیداری).",
        "• حذف زیرفضای پوچ: ker(∇_α Ŷ) = {0} (برخلاف دکودرهای غیرخطی، هیچ بردار عصبی نمی‌تواند توسط شبکه خنثی شود).",
        "• تابع هزینه جامع: L_total = -CC(Ŷ, Y) + 0.5·KLD(Ŷ, Y) + λ_ortho·L_ortho + λ_div·L_div",
        "• بهینه‌سازی محاسباتی: تنها ۴.۹۶ میلیون پارامتر (کاهش ۶۶.۸ درصدی حجم مدل) با آموزش ۴۶ دوره‌ای فوق‌پایدار."
    ]
    for pt in math_bullets:
        p = mb_tf.add_paragraph()
        p.text = pt
        p.font.size = Pt(12)
        p.font.color.rgb = COLOR_BODY_TEXT
        p.alignment = PP_ALIGN.RIGHT

    # -------------------------------------------------------------------------
    # LEFT COLUMN: Section 3 - نتایج (Results)
    # -------------------------------------------------------------------------
    create_card(l_col_x, Inches(4.7), col_w, Inches(19.2), "نتایج و تحلیل تجربی")
    s3_box = slide.shapes.add_textbox(l_col_x + Inches(0.4), Inches(5.5), col_w - Inches(0.8), Inches(3.6))
    s3_tf = s3_box.text_frame
    s3_tf.word_wrap = True
    s3_tf.margin_left = s3_tf.margin_top = s3_tf.margin_right = s3_tf.margin_bottom = 0

    results_text = [
        "ارزیابی جامع تجربی بر روی دادگان آزمون، برتری قاطع BrainGaze در وفاداری بازسازی و مهار فروپاشی مُدالیته را اثبات می‌کند:",
        "• دقت فراتر از وضعیت هنر: دستیابی به همبستگی شگفت‌انگیز CC = 0.8609، معیار شباهت SIM = 0.6970 و امتیاز NSS = 3.3032 با کاهش ۶۶.۸ درصدی پارامترها (تنها ۴.۹۶ میلیون پارامتر).",
        "• ممیزی تشخیصی استرس‌تست NVDS: در آزمون جایگزینی نویز گاوسی، مدل پیشنهادی حساسیت واقعی +24.65% را ثبت کرد؛ در حالی که مدل‌های مرجع ادبیات (Palazzo, Wang, Min, Kaushik) دچار غلبه مطلق تصویر بوده و حساسیت نویز نزدیک به صفر (+0.00% تا +0.10%) داشتند."
    ]
    for i, pt in enumerate(results_text):
        p = s3_tf.paragraphs[0] if i == 0 else s3_tf.add_paragraph()
        p.text = pt
        p.font.size = Pt(13.0)
        p.font.color.rgb = COLOR_BODY_TEXT
        p.alignment = PP_ALIGN.RIGHT

    # 1. Trio Figure in Section 3
    trio_img_path = os.path.join(FIGURES_DIR, "poster_trio_figure.png")
    if os.path.exists(trio_img_path):
        slide.shapes.add_picture(trio_img_path, l_col_x + Inches(0.4), Inches(9.2), width=col_w - Inches(0.8))

    # 2. Comparative Table in Section 3
    rows, cols = 6, 6
    tbl_left = l_col_x + Inches(0.4)
    tbl_top = Inches(18.7)
    tbl_w = col_w - Inches(0.8)
    tbl_h = Inches(4.8)

    table_shape = slide.shapes.add_table(rows, cols, tbl_left, tbl_top, tbl_w, tbl_h)
    tbl = table_shape.table

    tbl.columns[0].width = Inches(2.2)
    tbl.columns[1].width = Inches(1.3)
    tbl.columns[2].width = Inches(1.2)
    tbl.columns[3].width = Inches(1.5)
    tbl.columns[4].width = Inches(1.5)
    tbl.columns[5].width = Inches(2.1)

    table_data = [
        ["معماری / مطالعه", "نوع ورودی", "دقت (CC)", "حساسیت نویز", "تمایز سوژه", "وضعیت ممیزی NVDS"],
        ["Palazzo et al. (TPAMI 2021)", "EEG + ResNet", "MSE = 0.59", "+0.00%", "+0.00%", "❌ فروپاشی مُدالیته"],
        ["Wang et al. (CVPR 2020)", "EEG + ResNet", "MSE = 0.84", "+0.00%", "+0.00%", "❌ فروپاشی مُدالیته"],
        ["Min et al. (T-NSRE 2021)", "EEG + CNN", "0.6520", "+0.10%", "+0.02%", "❌ فروپاشی مُدالیته"],
        ["Kaushik et al. (NeuroImage)", "EEG + ResNet", "0.6840", "+0.05%", "+0.01%", "❌ فروپاشی مُدالیته"],
        ["★ BrainGaze (مدل پیشنهادی)", "EEG + ResNet", "0.8609", "+24.65%", "+18.75%", "✅ تأیید کامل NVDS"]
    ]

    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            cell = tbl.cell(r_idx, c_idx)
            cell.text = val
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            if r_idx == 0:
                p.font.bold = True
                p.font.size = Pt(11)
                p.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLOR_TABLE_HEADER
            elif r_idx == 5:
                p.font.bold = True
                p.font.size = Pt(11)
                p.font.color.rgb = COLOR_PASSED_GREEN
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(0xDC, 0xFC, 0xE7)
            else:
                p.font.size = Pt(10.5)
                p.font.color.rgb = COLOR_BODY_TEXT
                cell.fill.solid()
                cell.fill.fore_color.rgb = COLOR_TABLE_ROW_ALT if r_idx % 2 == 1 else RGBColor(0xFF, 0xFF, 0xFF)

    # -------------------------------------------------------------------------
    # LEFT COLUMN: Section 4 - جمع بندی (Conclusion & Applications)
    # -------------------------------------------------------------------------
    create_card(l_col_x, Inches(24.2), col_w, Inches(4.8), "جمع بندی")
    s4_box = slide.shapes.add_textbox(l_col_x + Inches(0.4), Inches(25.0), col_w - Inches(0.8), Inches(3.8))
    s4_tf = s4_box.text_frame
    s4_tf.word_wrap = True
    s4_tf.margin_left = s4_tf.margin_top = s4_tf.margin_right = s4_tf.margin_bottom = 0

    conclusion_points = [
        "در این پژوهش پدیده یادگیری میان‌بُر و غلبه تصویر در مدل‌های چندحالته عصبی-دیداری شناسایی و با تدوین استاندارد تشخیصی NVDS برای نخستین بار در ادبیات علمی ممیزی گردید. معماری BrainGaze با بهره‌گیری از مسیریابی پیمانه‌ای دوخطی و تضمین تعامد پارسوال، این چالش را به صورت قطعی مهار کرد.",
        "• محدودیت‌ها: وابستگی به کیفیت ثبت الکترودهای سطحی EEG و ضرورت حذف برخط آرتیفکت‌های حرکتی و چشمی.",
        "• کاربردهای صنعتی و بالینی:",
        "۱. واسط‌های مغز و رایانه (BCI) پسیو بدون نیاز به ردیاب‌های چشمی گران‌قیمت (Eye-Tracker).",
        "۲. رندرینگ متمرکز بر کانون توجه (Foveated Rendering) در هدست‌های واقعیت مجازی/افزوده (VR/AR).",
        "۳. غربالگری بالینی و عینی اختلالات شناختی، نقص توجه و بیش‌فعالی (ADHD) و اوتیسم (ASD)."
    ]
    for i, pt in enumerate(conclusion_points):
        p = s4_tf.paragraphs[0] if i == 0 else s4_tf.add_paragraph()
        p.text = pt
        p.font.size = Pt(12.5)
        p.font.color.rgb = COLOR_BODY_TEXT
        p.alignment = PP_ALIGN.RIGHT

    # -------------------------------------------------------------------------
    # LEFT COLUMN: Section 5 - مراجع اصلی (Main References)
    # -------------------------------------------------------------------------
    create_card(l_col_x, Inches(29.3), col_w, Inches(3.0), "مراجع اصلی")
    s5_box = slide.shapes.add_textbox(l_col_x + Inches(0.4), Inches(30.0), col_w - Inches(0.8), Inches(2.2))
    s5_tf = s5_box.text_frame
    s5_tf.word_wrap = True
    s5_tf.margin_left = s5_tf.margin_top = s5_tf.margin_right = s5_tf.margin_bottom = 0

    refs = [
        "[1] S. Palazzo et al., \"Decoding brain representations by multimodal learning of neural activity and visual features,\" IEEE TPAMI, vol. 43, no. 11, pp. 3833–3849, 2021.",
        "[2] W. Wang, D. Tran, and M. Feiszli, \"What makes training multi-modal networks hard?,\" in Proc. IEEE/CVF CVPR, 2020, pp. 12695–12704.",
        "[3] A. Kastrati et al., \"EEGEyeNet: A large-scale benchmark for simultaneous EEG and eye-tracking,\" in Adv. NeurIPS, vol. 34, 2021."
    ]
    for i, ref in enumerate(refs):
        p = s5_tf.paragraphs[0] if i == 0 else s5_tf.add_paragraph()
        p.text = ref
        p.font.size = Pt(10.5)
        p.font.color.rgb = COLOR_MUTED_TEXT
        p.alignment = PP_ALIGN.LEFT

    prs.save(OUTPUT_PPTX)
    print(f"Poster presentation successfully created and saved to:\n{OUTPUT_PPTX}")

if __name__ == "__main__":
    build_poster()
