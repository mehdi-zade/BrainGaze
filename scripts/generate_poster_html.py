"""
scripts/generate_poster_html.py
===============================
Builds the standalone, print-ready, high-resolution HTML poster (outputs/poster_presentation.html)
and exports it to a vector/high-res PDF (outputs/poster_presentation.pdf) using headless Microsoft Edge.
"""

import os
import subprocess
import base64
import pypdfium2 as pdfium

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
FIGURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "figures")
ASSETS_DIR = os.path.join(FIGURES_DIR, "template_assets")
OUTPUT_HTML = os.path.join(PROJECT_ROOT, "outputs", "poster_presentation.html")
OUTPUT_PDF  = os.path.join(PROJECT_ROOT, "outputs", "poster_presentation.pdf")
OUTPUT_PREVIEW = os.path.join(PROJECT_ROOT, "outputs", "poster_preview.png")

def get_base64_image(image_path):
    if not os.path.exists(image_path):
        return ""
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")
    if ext == "jpg": ext = "jpeg"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{ext};base64,{data}"

def generate_html_poster():
    print("Embedding assets into base64 for self-contained portability...")
    ut_logo_b64 = get_base64_image(os.path.join(ASSETS_DIR, "img_1_Image19.png"))
    ece_logo_b64 = get_base64_image(os.path.join(ASSETS_DIR, "img_2_Image21.png"))
    arch_diagram_b64 = get_base64_image(os.path.join(FIGURES_DIR, "poster_architecture_diagram.png"))
    trio_figure_b64 = get_base64_image(os.path.join(FIGURES_DIR, "poster_trio_figure.png"))

    html_template = """<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>پوستر دفاعیه پایان‌نامه کارشناسی: تلفیق چندحالته شناختی-دیداری (BrainGaze)</title>
    <!-- Google Fonts: Vazirmatn & Inter -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Vazirmatn:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --primary: #1a237e;
            --primary-dark: #0d1642;
            --accent-purple: #4a148c;
            --accent-purple-light: #7b1fa2;
            --card-border: #4a148c;
            --bg-page: #f8fafc;
            --bg-card: #ffffff;
            --text-main: #0f172a;
            --text-body: #1e293b;
            --text-muted: #475569;
            --badge-green-bg: #dcfce7;
            --badge-green-text: #15803d;
            --badge-red-bg: #fee2e2;
            --badge-red-text: #b91c1c;
            --font-family: 'Vazirmatn', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        @page {
            size: 594mm 841mm; /* Standard ISO A1 Portrait */
            margin: 0;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: var(--font-family);
            background-color: var(--bg-page);
            color: var(--text-body);
            line-height: 1.6;
            direction: rtl;
            -webkit-font-smoothing: antialiased;
        }

        .poster-container {
            width: 1240px;
            min-height: 1754px; /* Exact 1:1.414 aspect ratio */
            margin: 0 auto;
            background: #ffffff;
            padding: 38px 42px;
            display: flex;
            flex-direction: column;
            gap: 22px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
            position: relative;
        }

        /* -------------------------------------------------------------
           1. Header Styling
        ------------------------------------------------------------- */
        .poster-header {
            background: #ffffff;
            border: 3.5px solid var(--card-border);
            border-radius: 24px;
            padding: 22px 32px;
            display: grid;
            grid-template-columns: 140px 1fr 140px;
            align-items: center;
            gap: 20px;
            box-shadow: 0 4px 18px rgba(74, 20, 140, 0.06);
        }

        .logo-box {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .logo-box img {
            max-width: 100%;
            height: auto;
            max-height: 125px;
            object-fit: contain;
        }

        .header-center {
            text-align: center;
        }

        .header-title-label {
            font-size: 26.5px;
            font-weight: 900;
            color: var(--primary);
            margin-bottom: 6px;
            letter-spacing: -0.5px;
            line-height: 1.35;
        }

        .header-subtitle {
            font-size: 17px;
            font-weight: 700;
            color: var(--accent-purple);
            margin-bottom: 12px;
        }

        .header-meta {
            display: flex;
            justify-content: center;
            gap: 45px;
            font-size: 19px;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 6px;
        }

        .header-meta span b {
            color: var(--primary);
        }

        .header-affiliation {
            font-size: 16px;
            font-weight: 600;
            color: var(--text-muted);
        }

        /* -------------------------------------------------------------
           2. Two-Column Grid Layout
        ------------------------------------------------------------- */
        .poster-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            align-items: stretch;
        }

        .column {
            display: flex;
            flex-direction: column;
            gap: 22px;
        }

        /* -------------------------------------------------------------
           3. Card Components
        ------------------------------------------------------------- */
        .card {
            background: #ffffff;
            border: 2.8px solid var(--card-border);
            border-radius: 20px;
            padding: 22px 24px;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.03);
            display: flex;
            flex-direction: column;
        }

        .card-title {
            text-align: center;
            font-size: 23px;
            font-weight: 900;
            color: var(--card-border);
            margin-bottom: 16px;
            position: relative;
            padding-bottom: 10px;
        }

        .card-title::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 75px;
            height: 3.5px;
            background: linear-gradient(90deg, var(--card-border), var(--primary));
            border-radius: 3px;
        }

        .card-body {
            font-size: 14.2px;
            color: var(--text-body);
            text-align: justify;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .card-body p {
            margin-bottom: 4px;
            line-height: 1.65;
        }

        .bullet-list {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .bullet-list li {
            position: relative;
            padding-right: 20px;
            text-align: justify;
            line-height: 1.62;
        }

        .bullet-list li::before {
            content: '•';
            position: absolute;
            right: 0;
            color: var(--accent-purple);
            font-size: 20px;
            line-height: 1;
        }

        .bullet-list li b {
            color: var(--primary);
            font-weight: 700;
        }

        /* Technical Highlights / Formulation Box */
        .math-box {
            background: #f8fafc;
            border: 1.8px solid #cbd5e1;
            border-radius: 14px;
            padding: 12px 16px;
            margin-top: 10px;
            display: flex;
            flex-direction: column;
            gap: 6px;
            font-size: 13px;
        }

        .math-box-title {
            font-weight: 800;
            color: var(--primary);
            font-size: 13.5px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        /* -------------------------------------------------------------
           4. Figure Containers
        ------------------------------------------------------------- */
        .figure-container {
            margin-top: 12px;
            display: flex;
            flex-direction: column;
            align-items: center;
            background: #ffffff;
            border-radius: 14px;
            overflow: hidden;
            border: 1.5px solid #cbd5e1;
            padding: 6px;
        }

        .figure-container img {
            width: 100%;
            height: auto;
            border-radius: 10px;
            display: block;
        }

        .figure-caption {
            font-size: 11.5px;
            font-weight: 700;
            color: var(--text-muted);
            margin-top: 7px;
            text-align: center;
            direction: rtl;
        }

        /* -------------------------------------------------------------
           5. Comparative Table
        ------------------------------------------------------------- */
        .table-container {
            margin-top: 12px;
            overflow-x: auto;
            border-radius: 12px;
            border: 1.5px solid #cbd5e1;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
            text-align: center;
            background: #ffffff;
        }

        th {
            background: #1e293b;
            color: #ffffff;
            font-weight: 700;
            padding: 8px 6px;
            border-bottom: 2px solid #0f172a;
        }

        td {
            padding: 7px 6px;
            border-bottom: 1px solid #e2e8f0;
            color: var(--text-body);
        }

        tr:nth-child(even) td {
            background-color: #f8fafc;
        }

        tr.highlight-row td {
            background-color: #f0fdf4 !important;
            font-weight: 700;
            color: #15803d;
            border-top: 2px solid #16a34a;
            border-bottom: 2px solid #16a34a;
        }

        .badge-fail {
            display: inline-block;
            padding: 2px 7px;
            border-radius: 6px;
            font-size: 10.5px;
            font-weight: 700;
            background: var(--badge-red-bg);
            color: var(--badge-red-text);
        }

        .badge-pass {
            display: inline-block;
            padding: 2px 7px;
            border-radius: 6px;
            font-size: 10.5px;
            font-weight: 700;
            background: var(--badge-green-bg);
            color: var(--badge-green-text);
        }

        /* -------------------------------------------------------------
           6. References
        ------------------------------------------------------------- */
        .ref-list {
            list-style: none;
            display: flex;
            flex-direction: column;
            gap: 6px;
            font-size: 11.2px;
            color: var(--text-muted);
            direction: ltr;
            text-align: left;
        }

        .ref-list li {
            line-height: 1.45;
        }

        /* Print Specifics */
        @media print {
            body {
                background: #ffffff;
            }
            .poster-container {
                width: 100%;
                min-height: 100vh;
                box-shadow: none;
                padding: 18mm 18mm;
            }
        }
    </style>
</head>
<body>

<div class="poster-container">

    <!-- ================= 1. POSTER HEADER ================= -->
    <header class="poster-header">
        <!-- Right Logo: University of Tehran -->
        <div class="logo-box">
            <img src="__UT_LOGO__" alt="آرم دانشگاه تهران">
        </div>

        <!-- Center Details -->
        <div class="header-center">
            <h1 class="header-title-label">عنوان پروژه: تلفیق چندحالته ویژگی‌های ردیابی شناختی مغز و محرک‌های دیداری جهت پیش‌بینی توجه بصری (BrainGaze)</h1>
            <div class="header-subtitle">ممیزی تشخیصی و مهار قطعی پدیده «یادگیری میان‌بُر» با مسیریابی پیمانه‌ای شناختی-دیداری (CVMR)</div>
            <div class="header-meta">
                <span><b>دانشجو:</b> مهدی عبدالله‌زاده</span>
                <span><b>استاد راهنما:</b> دکتر محمدعلی اخائی</span>
            </div>
            <div class="header-affiliation">دانشکده مهندسی برق و کامپیوتر، پردیس دانشکده‌های فنی، دانشگاه تهران</div>
        </div>

        <!-- Left Logo: Faculty of Engineering / ECE -->
        <div class="logo-box">
            <img src="__ECE_LOGO__" alt="آرم پردیس دانشکده‌های فنی و دانشکده برق و کامپیوتر دانشگاه تهران">
        </div>
    </header>

    <!-- ================= 2. TWO-COLUMN POSTER BODY ================= -->
    <div class="poster-grid">

        <!-- ================= RIGHT COLUMN (COLUMN 1 in RTL) ================= -->
        <div class="column">
            
            <!-- SECTION 1: مقدمه / خلاصه (~100 words) -->
            <section class="card">
                <h2 class="card-title">مقدمه / خلاصه</h2>
                <div class="card-body">
                    <ul class="bullet-list">
                        <li><b>نوع پروژه:</b> تحقیقاتی کاربردی و شبیه‌سازی عمیق چندوجهی (<span dir="ltr">Deep Multimodal Learning</span>) همراه با واسط‌های مغز و رایانه (<span dir="ltr">BCI</span>).</li>
                        <li><b>اهداف پروژه:</b> پیش‌بینی و بازسازی نقشه توجه بصری فردی ناظر با تلفیق سیگنال‌های ۳۲ کاناله EEG و محرک‌های دیداری، و مهار پدیده «فروپاشی مُدالیته» (<span dir="ltr">Modality Collapse</span>).</li>
                        <li><b>سوال اصلی و فرضیه:</b> چرا مدل‌های عصبی-دیداری متداول علی‌رغم ادعای سنتز چندحالته، سیگنال مغزی را نادیده گرفته و به میان‌بُر تصویری پناه می‌برند؟ آیا می‌توان با طراحی معماری فاقد زیرفضای پوچ، بهره‌گیری از سیگنال مغز را از نظر ریاضی گریزناپذیر کرد؟</li>
                        <li><b>روش پاسخ:</b> تدوین «استاندارد تشخیصی عصبی-بینایی» (<span dir="ltr">NVDS</span>) برای ممیزی رخنه‌گاه‌های FiLM و گیت‌های پسماند، و ابداع معماری نوین BrainGaze بر پایه مسیریابی پیمانه‌ای شناختی-دیداری (<span dir="ltr">CVMR</span>).</li>
                        <li><b>دستاوردها:</b> مهار قطعی میان‌بُر، ارتقای همبستگی به <b dir="ltr">CC = 0.8609</b> و دستیابی به حساسیت زیستی فعال <b dir="ltr">+24.65%</b> در برابر نویز مغزی.</li>
                    </ul>
                </div>
            </section>

            <!-- SECTION 2: روش/ساختار/مدل پیشنهادی (~120 words + Diagram + Math Box) -->
            <section class="card" style="flex: 1;">
                <h2 class="card-title">روش/ساختار/مدل پیشنهادی</h2>
                <div class="card-body">
                    <p>معماری تحول‌آفرین <b>BrainGaze (CVMR)</b> بر پایه تقسیم کار دوخطی و حذف ساختاری زیرفضای پوچ دکودر بنا شده است:</p>
                    <ul class="bullet-list">
                        <li><b>۱. شاخه دیداری (<span dir="ltr">Visual Stream</span>):</b> استخراج ویژگی مکانی با ResNet-18 منجمد و تولید <span dir="ltr">K=8</span> نقشه پایه مکانی متعامد (<span dir="ltr">M₁ ... M₈</span>) با مهار شباهت کسینوسی ماتریس گرام طبق قضیه تعامد پارسوال (<span dir="ltr">L_ortho</span>).</li>
                        <li><b>۲. شاخه شناختی مغز (<span dir="ltr">Cognitive Router</span>):</b> عبور سیگنال ۳۲ کاناله EEG از کانولوشن زمانی (<span dir="ltr">k=15</span>) و ترانسفورمر مکانی-زمانی به همراه بردار هویت سوژه، جهت تعیین وزن‌های مسیریابی <span dir="ltr" style="font-weight: 700; color: #1a237e;">α ∈ Δ⁸</span> با حداکثر آنتروپی (<span dir="ltr">L_div</span>).</li>
                        <li><b>۳. تلفیق دوخطی پیمانه‌ای:</b> بازسازی نهایی <span dir="ltr" style="font-weight: 700; color: #1a237e;">Ŷ = ∑ αₖ · Mₖ</span>؛ طبق <b>قضیه تعبیه هم‌اندازه پارسوال</b> (<span dir="ltr" style="font-weight: 700; color: #1a237e;">||ΔŶ||_L² ≡ ||Δα||₂</span>)، هر تغییر در وضعیت شناختی مغز الزاماً نقشه توجه را جابجا کرده و دور زدن سیگنال مغزی از نظر ریاضی ناممکن است.</li>
                        <li><b>محیط پیاده‌سازی و دادگان:</b> پیاده‌سازی در PyTorch 2.0 بر روی دادگان جامع BGD (شامل ۳,۲۳۴ نمونه همگام‌سازی‌شده EEG ناظران و تصاویر MS COCO).</li>
                    </ul>

                    <!-- Architecture Diagram -->
                    <div class="figure-container">
                        <img src="__ARCH_DIAGRAM__" alt="دیاگرام معماری نهایی مدل BrainGaze CVMR">
                        <div class="figure-caption">شکل ۱: دیاگرام جریان داده و معماری پیشنهادی BrainGaze: مسیریابی پیمانه‌ای شناختی-دیداری (CVMR) و اصل ایزومتری پارسوال</div>
                    </div>

                    <!-- Mathematical Proof & Specifications Box -->
                    <div class="math-box">
                        <div class="math-box-title">★ مبانی ریاضی و حذف رخنه‌گاه زیرفضای پوچ دکودر</div>
                        <div>• <b>تضمین هم‌اندازگی پارسوال:</b> <span dir="ltr" style="font-family: monospace; font-weight: bold; color: #1a237e;">||Ŷ₁ - Ŷ₂||_L² ≡ ||α₁ - α₂||₂</span> (تطابق کامل متریک اقلیدسی مغز و برجستگی دیداری).</div>
                        <div>• <b>حذف زیرفضای پوچ:</b> <span dir="ltr" style="font-family: monospace; font-weight: bold; color: #1a237e;">ker(∇_α Ŷ) = {0}</span> (برخلاف دکودرهای پیوسته، هیچ بردار عصبی خنثی یا صفر نمی‌شود).</div>
                        <div>• <b>فرمولاسیون تابع هزینه جامع:</b> <span dir="ltr" style="font-family: monospace; font-weight: bold; color: #1a237e;">L_total = -CC(Ŷ, Y) + 0.5·KLD(Ŷ, Y) + λ_ortho·L_ortho + λ_div·L_div</span></div>
                        <div>• <b>بهینه‌سازی محاسباتی:</b> تنها ۴.۹۶ میلیون پارامتر (کاهش ۶۶.۸ درصدی حجم مدل نسبت به رقبا) با آموزش ۴۶ دوره‌ای پایدار.</div>
                    </div>
                </div>
            </section>

        </div>

        <!-- ================= LEFT COLUMN (COLUMN 2 in RTL) ================= -->
        <div class="column">
            
            <!-- SECTION 3: نتایج (~120 words + Trio Figure + Comparison Table) -->
            <section class="card">
                <h2 class="card-title">نتایج و تحلیل تجربی</h2>
                <div class="card-body">
                    <p>ارزیابی‌های جامع تجربی بر روی داده‌های آزمون، برتری قاطع مدل پیشنهادی BrainGaze در بازسازی توجه و مهار فروپاشی مُدالیته را نشان می‌دهد:</p>
                    <ul class="bullet-list">
                        <li><b>دقت و وفاداری بالا:</b> دستیابی به رکورد همبستگی <b>CC = 0.8609</b>، شباهت توزیع <b>SIM = 0.6970</b> و امتیاز <b>NSS = 3.3032</b> با کاهش ۶۶.۸ درصدی پارامترهای محاسباتی (تنها ۴.۹۶ میلیون پارامتر).</li>
                        <li><b>ممیزی تشخیصی استاندارد NVDS:</b> در آزمون تزریق نویز گاوسی به سیگنال مغز، مدل پیشنهادی حساسیت زیستی فعال <b>+۲۴.۶۵٪</b> را ثبت کرد؛ در حالی که ۴ مدل مرجع بین‌المللی ادبیات (Palazzo, Wang, Min, Kaushik) حساسیت نزدیک به صفر (+0.00% تا +0.10%) داشته و دچار غلبه کامل تصویر هستند.</li>
                    </ul>

                    <!-- 1. Trio Figure (Requested by User) -->
                    <div class="figure-container">
                        <img src="__TRIO_FIGURE__" alt="نمونه‌های سه‌گانه تصویر، حقیقت زمینی و بازسازی">
                        <div class="figure-caption">شکل ۲: مقایسه تصویری ۳ نمونه سه‌گانه: تصویر محرک، حقیقت زمینی نگاه انسان (Ground Truth) و بازسازی مدل پیشنهادی BrainGaze</div>
                    </div>

                    <!-- 2. Benchmark Comparison Table (Requested by User) -->
                    <div class="table-container">
                        <table>
                            <thead>
                                <tr>
                                    <th>معماری / مطالعه</th>
                                    <th>نوع ورودی</th>
                                    <th>همبستگی (CC)</th>
                                    <th>حساسیت نویز</th>
                                    <th>تمایز سوژه</th>
                                    <th>ممیزی NVDS</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>Palazzo et al. (TPAMI 2021)</td>
                                    <td>EEG + ResNet</td>
                                    <td>MSE = 0.59</td>
                                    <td>+0.00%</td>
                                    <td>+0.00%</td>
                                    <td><span class="badge-fail">فروپاشی مُدالیته</span></td>
                                </tr>
                                <tr>
                                    <td>Wang et al. (CVPR 2020)</td>
                                    <td>EEG + ResNet</td>
                                    <td>MSE = 0.84</td>
                                    <td>+0.00%</td>
                                    <td>+0.00%</td>
                                    <td><span class="badge-fail">فروپاشی مُدالیته</span></td>
                                </tr>
                                <tr>
                                    <td>Min et al. (T-NSRE 2021)</td>
                                    <td>EEG + CNN</td>
                                    <td>0.6520</td>
                                    <td>+0.10%</td>
                                    <td>+0.02%</td>
                                    <td><span class="badge-fail">فروپاشی مُدالیته</span></td>
                                </tr>
                                <tr>
                                    <td>Kaushik et al. (NeuroImage)</td>
                                    <td>EEG + ResNet</td>
                                    <td>0.6840</td>
                                    <td>+0.05%</td>
                                    <td>+0.01%</td>
                                    <td><span class="badge-fail">فروپاشی مُدالیته</span></td>
                                </tr>
                                <tr>
                                    <td>EEGEyeNet (NeurIPS 2021)</td>
                                    <td>تک‌حالته (EEG)</td>
                                    <td>Err = 45px</td>
                                    <td>+108.7%</td>
                                    <td>+14.8%</td>
                                    <td><span class="badge-pass">حساسیت خالص</span></td>
                                </tr>
                                <tr class="highlight-row">
                                    <td>★ BrainGaze (مدل پیشنهادی)</td>
                                    <td>EEG + ResNet</td>
                                    <td>0.8609</td>
                                    <td>+24.65%</td>
                                    <td>+18.75%</td>
                                    <td><span class="badge-pass">✅ تأیید کامل NVDS</span></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </section>

            <!-- SECTION 4: جمع بندی (~100 words) -->
            <section class="card">
                <h2 class="card-title">جمع بندی</h2>
                <div class="card-body">
                    <p>در این پژوهش، معضل یادگیری میان‌بُر و غلبه تصویر در مدل‌های چندحالته توجه دیداری کالبدشکافی و با تدوین استاندارد تشخیصی NVDS برای نخستین بار اثبات شد که مدل‌های موجود دچار فروپاشی مدالیته هستند. مدل تحول‌آفرین BrainGaze با بهره‌گیری از تقسیم کار دوخطی و تضمین تعامد پارسوال، این چالش را به صورت قطعی حل کرد.</p>
                    <ul class="bullet-list">
                        <li><b>محدودیت‌ها:</b> وابستگی به کیفیت سیگنال ۳۲ کاناله و ضرورت پایش آرتیفکت‌های حرکتی در ثبت‌های میدانی.</li>
                        <li><b>کاربردهای صنعتی و بالینی:</b>
                            <br>• واسط‌های مغز و رایانه (BCI) پسیو بدون نیاز به سخت‌افزارهای گران‌قیمت ردیاب چشمی (Eye-Tracker).
                            <br>• بهینه‌سازی محاسباتی و رندرینگ متمرکز بر نگاه کاربر (Foveated Rendering) در هدست‌های واقعیت مجازی/افزوده (VR/AR).
                            <br>• ابزار غربالگری عینی و نورولوژیک اختلالات شناختی، نقص توجه و بیش‌فعالی (ADHD) و اوتیسم (ASD).
                        </li>
                    </ul>
                </div>
            </section>

            <!-- SECTION 5: مراجع اصلی (IEEE Standard) -->
            <section class="card">
                <h2 class="card-title">مراجع اصلی</h2>
                <div class="card-body">
                    <ul class="ref-list">
                        <li>[1] S. Palazzo et al., "Decoding brain representations by multimodal learning of neural activity and visual features," <i>IEEE Trans. Pattern Anal. Mach. Intell. (TPAMI)</i>, vol. 43, no. 11, pp. 3833–3849, 2021.</li>
                        <li>[2] W. Wang, D. Tran, and M. Feiszli, "What makes training multi-modal networks hard?," in <i>Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2020, pp. 12695–12704.</li>
                        <li>[3] A. Kastrati et al., "EEGEyeNet: A large-scale dataset and benchmark for simultaneous EEG and eye-tracking," in <i>Adv. Neural Inf. Process. Syst. (NeurIPS)</i>, vol. 34, 2021.</li>
                    </ul>
                </div>
            </section>

        </div>

    </div>

</div>

</body>
</html>
"""
    # Replace image placeholders
    html_content = html_template.replace("__UT_LOGO__", ut_logo_b64)
    html_content = html_content.replace("__ECE_LOGO__", ece_logo_b64)
    html_content = html_content.replace("__ARCH_DIAGRAM__", arch_diagram_b64)
    html_content = html_content.replace("__TRIO_FIGURE__", trio_figure_b64)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Standalone HTML poster successfully written to:\n{OUTPUT_HTML}")

def export_html_to_pdf():
    edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if not os.path.exists(edge_exe):
        print("Microsoft Edge executable not found, skipping PDF generation.")
        return

    print("Exporting HTML poster to high-resolution vector PDF using headless Edge...")
    cmd = [
        edge_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={OUTPUT_PDF}",
        OUTPUT_HTML
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(OUTPUT_PDF) and os.path.getsize(OUTPUT_PDF) > 0:
        print(f"Successfully exported print-ready PDF to:\n{OUTPUT_PDF} (Size: {os.path.getsize(OUTPUT_PDF):,} bytes)")
    else:
        print(f"Edge print-to-pdf completed. Output status: {res.stderr}")

def render_preview_image():
    if not os.path.exists(OUTPUT_PDF):
        print("PDF file not found, skipping preview image rendering.")
        return
    print("Rendering updated high-res preview image with pypdfium2...")
    pdf = pdfium.PdfDocument(OUTPUT_PDF)
    page = pdf[0]
    image = page.render(scale=1.5).to_pil()
    image.save(OUTPUT_PREVIEW)
    print(f"Updated preview image saved to: {OUTPUT_PREVIEW} ({image.size[0]}x{image.size[1]})")

if __name__ == "__main__":
    generate_html_poster()
    export_html_to_pdf()
    render_preview_image()
