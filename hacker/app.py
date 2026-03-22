from flask import Flask, jsonify, render_template, send_file, request
from flask_cors import CORS
from functools import lru_cache
import io
import os
import re
from datetime import datetime

from ml_engine import analyze_area, analyze_city_overview, get_cities, get_areas

# Tell Flask exact path to templates — fixes TemplateNotFound on Windows
BASE = os.path.dirname(os.path.abspath(__file__))
app  = Flask(__name__, template_folder=os.path.join(BASE, 'templates'))
CORS(app)


# ── Cache ML results — same area loads instantly second time ──
@lru_cache(maxsize=100)
def cached_analyze(city, area):
    return analyze_area(city, area)


# ── Auto-refresh cache at midnight every night ──
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    import atexit

    def refresh_data():
        print("\n🔄 Midnight refresh — clearing cached ML results...")
        cached_analyze.cache_clear()
        print("✅ Cache cleared — fresh satellite data loads on next request\n")

    scheduler = BackgroundScheduler()
    scheduler.add_job(func=refresh_data, trigger='cron', hour=0, minute=0)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
    print("⏰ Daily midnight refresh scheduled")

except ImportError:
    print("⚠️  apscheduler not installed — no auto-refresh (run: pip install apscheduler)")


# ── Fix area name mismatches between home.html and ml_engine ──
NAME_MAP = {
    'Narangpura': 'Naranpura',
    'Ambavadi':   'Ambawadi',
}


# ══════════════════════════════════════════════
#  PAGE ROUTES
# ══════════════════════════════════════════════

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# ══════════════════════════════════════════════
#  DATA API ROUTES
# ══════════════════════════════════════════════

@app.route('/api/cities')
def cities():
    return jsonify({'cities': get_cities()})


@app.route('/api/areas/<city>')
def areas(city):
    area_list = get_areas(city)
    if not area_list:
        return jsonify({
            'error': f'{city} not found. Only Ahmedabad and Gandhinagar supported.'
        }), 404
    return jsonify({'areas': area_list})


@app.route('/api/analyze/<city>/<area>')
def analyze(city, area):
    area = area.replace('%20', ' ').replace('+', ' ')
    area = NAME_MAP.get(area, area)
    try:
        result = cached_analyze(city, area)
        if result is None:
            return jsonify({'error': f'{area} not found in {city}'}), 404
        return jsonify(result)
    except KeyError:
        return jsonify({
            'error': f'{area} not found in {city}. Check spelling.'
        }), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/overview/<city>')
def overview(city):
    try:
        return jsonify(analyze_city_overview(city))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health')
def health():
    return jsonify({
        'status':  'running',
        'cities':  get_cities(),
        'message': 'EcoSentinel is live',
        'routes': {
            '/':                    'Landing page (GreenGrid map)',
            '/dashboard':           'ML Dashboard (?city=X&area=Y)',
            '/api/analyze/City/Area': 'Run ML models',
            '/api/advice/City/Area':  'Get AI action plan',
            '/api/report/City/Area':  'Download PDF report',
        }
    })


# ── Manual cache refresh endpoint ──
@app.route('/api/refresh')
def refresh():
    cached_analyze.cache_clear()
    return jsonify({'status': 'Cache cleared — fresh data loads on next request'})


# ══════════════════════════════════════════════
#  AI ADVICE FUNCTION
# ══════════════════════════════════════════════

def get_ai_advice(data):
    try:
        import anthropic
        client = anthropic.Anthropic()
        prompt = f"""You are an environmental advisor for Indian municipal corporations.
Analyze this satellite ML data for {data['area']}, {data['city']} and give practical advice.

AREA DATA:
- Area type: {data['type']}
- Average temperature: {data['avg_temp']}°C
- Average NO2 pollution: {data['avg_no2']} μg/m³
- Unusual temperature days: {data['anomaly_count']} days
- Worst temperature: {data.get('worst_anomaly_temp', 'N/A')}°C
- 30-day forecast: {data['forecast_avg']}°C (trend: {data['trend']})
- Pollution zones: {data['hotspot_clusters']}
- Habitability: {data['habitability']['score']}/100 ({data['habitability']['label']})
- Risk: {data['risk']['label']} ({data['risk']['score']}/100)
- Data period: {data.get('data_period', '2023')}

Give your response in these sections:
1. CURRENT SITUATION (2 lines plain English)
2. IMMEDIATE ACTIONS NEEDED (3-4 specific actions)
3. 30-DAY ACTION PLAN (Week 1 / Week 2-3 / Week 4)
4. WHO SHOULD ACT
5. EXPECTED IMPROVEMENT

Maximum 300 words. Simple language. No jargon."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    except Exception:
        # Rule-based fallback when Claude API unavailable
        risk = data['risk']['score']
        area = data['area']
        temp = data['avg_temp']
        no2  = data['avg_no2']
        hab  = data['habitability']['score']

        if risk > 70:
            return f"""1. CURRENT SITUATION
{area} is in HIGH RISK state. Temperature of {temp}°C with NO2 at {no2} ug/m3 requires immediate government action. {data['anomaly_count']} dangerous temperature days were detected.

2. IMMEDIATE ACTIONS NEEDED
- Restrict factory operations between 10am and 6pm daily
- Issue public health advisory for outdoor activity during peak hours
- Deploy mobile air quality monitoring units across affected zones
- Activate emergency tree-watering and dust suppression protocols

3. 30-DAY ACTION PLAN
Week 1: Survey top 10 pollution sources. Issue formal notices to industrial units.
Week 2-3: Plant 500+ saplings. Install dust barriers on construction sites.
Week 4: Re-measure air quality. Submit compliance report to AMC Environment Dept.

4. WHO SHOULD ACT
AMC Environment Department + Gujarat Pollution Control Board (GPCB) + Local factory owners.

5. EXPECTED IMPROVEMENT
NO2 can reduce 20-30% in 90 days. Habitability can improve from {hab} to above 60 in 3 months."""

        elif risk > 40:
            return f"""1. CURRENT SITUATION
{area} is at MODERATE RISK. NO2 at {no2} ug/m3 exceeds the safe limit of 40 ug/m3. Monitoring required.

2. IMMEDIATE ACTIONS NEEDED
- Plant trees along all main roads and intersections
- Set up 2 permanent air quality monitoring stations
- Enforce vehicle pollution checks at key intersections
- Promote rooftop gardens in housing societies

3. 30-DAY ACTION PLAN
Week 1: Install monitors, record baseline measurements.
Week 2-3: Tree plantation drive with Resident Welfare Associations.
Week 4: Review data and publish public environmental status report.

4. WHO SHOULD ACT
Ward-level AMC officers + Resident Welfare Associations + Local industries.

5. EXPECTED IMPROVEMENT
Habitability can improve from {hab} to above 70 within 60 days of action."""

        else:
            return f"""1. CURRENT SITUATION
{area} has GOOD environmental conditions. Habitability of {hab}/100 is commendable. Focus on sustaining this.

2. HOW TO MAINTAIN
- Continue green cover maintenance programs
- Monitor NO2 levels monthly and publish publicly
- Prevent new heavy industries from setting up
- Share best practices with neighbouring areas

3. 30-DAY ACTION PLAN
Week 1: Document current green cover with satellite data.
Week 2-3: Community awareness programs about clean air.
Week 4: Designate area as Model Environmental Zone.

4. WHO SHOULD ACT
Local ward committee + community volunteers + schools + municipal garden department.

5. EXPECTED IMPROVEMENT
Sustain current {hab}/100 score and target 85+ by year end."""


@app.route('/api/advice/<city>/<area>')
def get_advice_api(city, area):
    area = area.replace('%20', ' ').replace('+', ' ')
    area = NAME_MAP.get(area, area)
    try:
        data   = cached_analyze(city, area)
        advice = get_ai_advice(data)
        return jsonify({'advice': advice})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════
#  PROFESSIONAL PDF REPORT
# ══════════════════════════════════════════════

@app.route('/api/report/<city>/<area>')
def generate_report(city, area):
    area = area.replace('%20', ' ').replace('+', ' ')
    area = NAME_MAP.get(area, area)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, KeepTogether
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY

        data        = cached_analyze(city, area)
        risk        = data.get('risk', {})
        hab         = data.get('habitability', {})
        today       = datetime.now().strftime("%d %B %Y")
        ref_no      = f"GG/{city[:3].upper()}/{datetime.now().strftime('%Y%m%d')}/001"
        data_period = data.get('data_period', '2023')

        # --- Colours ---
        DARK_NAVY   = colors.HexColor('#0D1B2A')
        NAVY        = colors.HexColor('#1B2A4A')
        TEAL        = colors.HexColor('#0D6E6E')
        TEAL_LIGHT  = colors.HexColor('#E8F5F5')
        GOLD        = colors.HexColor('#C49A2C')
        GOLD_LIGHT  = colors.HexColor('#FDF6E3')
        WHITE       = colors.white
        LIGHT_GRAY  = colors.HexColor('#F5F6F7')
        MID_GRAY    = colors.HexColor('#E2E4E8')
        DARK_GRAY   = colors.HexColor('#4A5568')
        BLACK       = colors.HexColor('#1A202C')
        RED         = colors.HexColor('#C53030')
        ORANGE      = colors.HexColor('#C05621')
        GREEN_GOOD  = colors.HexColor('#276749')
        TEAL_MID    = colors.HexColor('#0D6E6E')

        def rc(s):
            return RED if s > 70 else ORANGE if s > 40 else GREEN_GOOD

        def no2_status(v):
            if v > 80:   return ("Hazardous",  RED)
            elif v > 50: return ("Unhealthy",  ORANGE)
            elif v > 30: return ("Moderate",   GOLD)
            else:        return ("Good",       GREEN_GOOD)

        def temp_status(v):
            if v > 35:   return ("Critical",  RED)
            elif v > 28: return ("Elevated",  ORANGE)
            else:        return ("Normal",    GREEN_GOOD)

        # --- Style helper — unique name each call to avoid ReportLab conflicts ---
        _sn = [0]
        def S(**kw):
            _sn[0] += 1
            name = f"s{_sn[0]}"
            return ParagraphStyle(name, **kw)

        # Base styles
        s_section  = S(fontName='Helvetica-Bold', fontSize=12, textColor=TEAL,
                       leading=18, spaceBefore=16, spaceAfter=8)
        s_body     = S(fontName='Helvetica', fontSize=10, textColor=DARK_GRAY,
                       leading=16, spaceAfter=6, alignment=TA_JUSTIFY)
        s_body_b   = S(fontName='Helvetica-Bold', fontSize=10, textColor=BLACK,
                       leading=16, spaceAfter=6)
        s_footer   = S(fontName='Helvetica', fontSize=8,
                       textColor=colors.HexColor('#94A3B8'), alignment=TA_CENTER)
        s_th       = S(fontName='Helvetica-Bold', fontSize=9, textColor=WHITE,
                       leading=13, alignment=TA_LEFT)
        s_tc       = S(fontName='Helvetica', fontSize=9, textColor=DARK_GRAY,
                       leading=14, alignment=TA_LEFT)
        s_tc_b     = S(fontName='Helvetica-Bold', fontSize=9, textColor=BLACK,
                       leading=14, alignment=TA_LEFT)
        s_note     = S(fontName='Helvetica-Oblique', fontSize=8,
                       textColor=colors.HexColor('#6B8A7A'),
                       leading=12, leftIndent=6, spaceAfter=4)
        s_adv_h    = S(fontName='Helvetica-Bold', fontSize=10, textColor=NAVY,
                       leading=14, spaceBefore=10, spaceAfter=3)
        s_adv_b    = S(fontName='Helvetica', fontSize=10, textColor=DARK_GRAY,
                       leading=15, spaceAfter=4, leftIndent=10, alignment=TA_JUSTIFY)

        # --- Document setup ---
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=1.5*cm, bottomMargin=2*cm,
            title=f"GreenGrid Report - {area}, {city}",
            author="GreenGrid Environmental Intelligence"
        )
        W   = 17 * cm
        cnt = []

        # ══════════════════════
        #  HEADER BANNER
        # ══════════════════════
        banner = Table([[Paragraph(
            'GREENGRID  -  SATELLITE ENVIRONMENTAL INTELLIGENCE PLATFORM  -  CONFIDENTIAL',
            S(fontName='Helvetica-Bold', fontSize=8,
              textColor=colors.HexColor('#A0B4C8'), alignment=TA_CENTER)
        )]], colWidths=[W])
        banner.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), DARK_NAVY),
            ('TOPPADDING',    (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('LINEBELOW', (0,0), (-1,-1), 3, GOLD),
        ]))
        cnt.append(banner)
        cnt.append(Spacer(1, 0.6*cm))

        # ══════════════════════
        #  LOGO ROW
        # ══════════════════════
        logo_row = Table([[
            Paragraph('<b>GREENGRID</b>',
                S(fontName='Helvetica-Bold', fontSize=20, textColor=TEAL)),
            Paragraph(
                'Powered by NASA POWER  |  Isolation Forest  |  ARIMA  |  DBSCAN',
                S(fontName='Helvetica', fontSize=8,
                  textColor=colors.HexColor('#94A3B8'), alignment=TA_RIGHT)),
        ]], colWidths=[W*0.45, W*0.55])
        logo_row.setStyle(TableStyle([
            ('VALIGN',         (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING',     (0,0), (-1,-1), 0),
            ('BOTTOMPADDING',  (0,0), (-1,-1), 0),
        ]))
        cnt.append(logo_row)
        cnt.append(HRFlowable(width=W, thickness=1, color=MID_GRAY, spaceAfter=12))

        # ══════════════════════
        #  TITLE BLOCK
        # ══════════════════════
        cnt.append(Paragraph(
            'ENVIRONMENTAL INTELLIGENCE REPORT',
            S(fontName='Helvetica-Bold', fontSize=9, textColor=GOLD, leading=12)))
        cnt.append(Spacer(1, 0.2*cm))
        cnt.append(Paragraph(
            area.upper(),
            S(fontName='Helvetica-Bold', fontSize=26, textColor=DARK_NAVY, leading=30)))
        cnt.append(Paragraph(
            f'{city}, Gujarat, India',
            S(fontName='Helvetica-Bold', fontSize=13, textColor=TEAL, leading=18)))
        cnt.append(Spacer(1, 0.3*cm))
        cnt.append(Paragraph(
            'This report analyses real satellite data to show the environmental '
            'health of this area. It is designed to be read by government officials, '
            'municipal commissioners, and also by the general public.',
            s_body))
        cnt.append(Spacer(1, 0.4*cm))

        # ══════════════════════
        #  META INFO TABLE
        # ══════════════════════
        meta = Table([[
            Paragraph(f'<b>Reference No.</b><br/>{ref_no}',
                S(fontName='Helvetica', fontSize=9, textColor=DARK_GRAY, leading=14)),
            Paragraph(f'<b>Report Date</b><br/>{today}',
                S(fontName='Helvetica', fontSize=9, textColor=DARK_GRAY, leading=14)),
            Paragraph(f'<b>Data Period</b><br/>{data_period}',
                S(fontName='Helvetica', fontSize=9, textColor=DARK_GRAY, leading=14)),
            Paragraph(f'<b>Area Type</b><br/>{data.get("type","mixed").title()}',
                S(fontName='Helvetica', fontSize=9, textColor=DARK_GRAY, leading=14)),
        ]], colWidths=[W/4]*4)
        meta.setStyle(TableStyle([
            ('BOX',        (0,0), (-1,-1), 1,   MID_GRAY),
            ('INNERGRID',  (0,0), (-1,-1), 0.5, MID_GRAY),
            ('TOPPADDING',    (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('LEFTPADDING',   (0,0), (-1,-1), 10),
            ('RIGHTPADDING',  (0,0), (-1,-1), 10),
            ('VALIGN',     (0,0), (-1,-1), 'TOP'),
        ]))
        cnt.append(meta)
        cnt.append(Spacer(1, 0.5*cm))

        # ══════════════════════
        #  4 KEY METRIC CARDS
        #  FLAT TABLE — no nested tables (fixes overlap)
        # ══════════════════════
        rs = risk.get('score', 0);   rl = risk.get('label', '--')
        hs = hab.get('score', 0);    hl = hab.get('label', '--')
        tstat, tcol = temp_status(data.get('avg_temp', 0))
        nstat, ncol = no2_status(data.get('avg_no2', 0))

        # Row 1 = titles, Row 2 = big values, Row 3 = status labels
        card_data = [
            # --- row 1: titles ---
            [
                Paragraph('OVERALL RISK SCORE',
                    S(fontName='Helvetica-Bold', fontSize=8,
                      textColor=colors.HexColor('#64748B'), leading=11)),
                Paragraph('HABITABILITY INDEX',
                    S(fontName='Helvetica-Bold', fontSize=8,
                      textColor=colors.HexColor('#64748B'), leading=11)),
                Paragraph('AVG TEMPERATURE',
                    S(fontName='Helvetica-Bold', fontSize=8,
                      textColor=colors.HexColor('#64748B'), leading=11)),
                Paragraph('AVG AIR POLLUTION (NO2)',
                    S(fontName='Helvetica-Bold', fontSize=8,
                      textColor=colors.HexColor('#64748B'), leading=11)),
            ],
            # --- row 2: big values ---
            [
                Paragraph(f'<font size="24" color="{rc(rs).hexval()}"><b>{rs}</b></font>'
                          '<font size="11" color="#64748B"> / 100</font>',
                    S(fontName='Helvetica', fontSize=24, leading=30)),
                Paragraph(f'<font size="24" color="{(GREEN_GOOD if hs>60 else ORANGE if hs>40 else RED).hexval()}"><b>{hs}</b></font>'
                          '<font size="11" color="#64748B"> / 100</font>',
                    S(fontName='Helvetica', fontSize=24, leading=30)),
                Paragraph(f'<font size="24" color="{tcol.hexval()}"><b>{data.get("avg_temp","--")}</b></font>'
                          '<font size="11" color="#64748B"> degC</font>',
                    S(fontName='Helvetica', fontSize=24, leading=30)),
                Paragraph(f'<font size="24" color="{ncol.hexval()}"><b>{data.get("avg_no2","--")}</b></font>'
                          '<font size="11" color="#64748B"> ug/m3</font>',
                    S(fontName='Helvetica', fontSize=24, leading=30)),
            ],
            # --- row 3: status labels ---
            [
                Paragraph(rl,  S(fontName='Helvetica-Bold', fontSize=9,
                                  textColor=rc(rs), leading=12)),
                Paragraph(hl,  S(fontName='Helvetica-Bold', fontSize=9,
                                  textColor=GREEN_GOOD if hs>60 else ORANGE if hs>40 else RED,
                                  leading=12)),
                Paragraph(tstat, S(fontName='Helvetica-Bold', fontSize=9,
                                    textColor=tcol, leading=12)),
                Paragraph(nstat, S(fontName='Helvetica-Bold', fontSize=9,
                                    textColor=ncol, leading=12)),
            ],
        ]

        cards = Table(card_data, colWidths=[W/4]*4)
        cards.setStyle(TableStyle([
            ('BOX',           (0,0), (-1,-1), 1,   MID_GRAY),
            ('INNERGRID',     (0,0), (-1,-1), 0.5, MID_GRAY),
            ('TOPPADDING',    (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('LEFTPADDING',   (0,0), (-1,-1), 12),
            ('RIGHTPADDING',  (0,0), (-1,-1), 12),
            ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
            ('LINEBELOW',     (0,2), (-1,2),   3,   TEAL),
            ('BACKGROUND',    (0,0), (-1,0),   LIGHT_GRAY),
        ]))
        cnt.append(cards)
        cnt.append(Spacer(1, 0.6*cm))

        # ══════════════════════
        #  SECTION 1 — WHAT THIS MEANS (plain language first)
        # ══════════════════════
        cnt.append(HRFlowable(width=W, thickness=2, color=TEAL, spaceAfter=8))
        cnt.append(Paragraph('1. WHAT THIS MEANS — SIMPLE EXPLANATION', s_section))

        # Plain language summary box
        plain_lines = [
            Paragraph('<b>For common people — in simple words:</b>',
                S(fontName='Helvetica-Bold', fontSize=10, textColor=TEAL, leading=14)),
            Spacer(1, 0.2*cm),
        ]
        avg_t = data.get('avg_temp', 30)
        avg_n = data.get('avg_no2', 50)
        a_cnt = data.get('anomaly_count', 0)
        f_avg = data.get('forecast_avg', 30)
        trend = data.get('trend', 'Stable')

        if rs > 70:
            plain_lines.append(Paragraph(
                f'{area} is in a DANGEROUS condition right now. The air is polluted and '
                f'temperatures are very high. This is not good for people\'s health, '
                f'especially for children and elderly people.',
                S(fontName='Helvetica', fontSize=10, textColor=BLACK, leading=16)))
        elif rs > 40:
            plain_lines.append(Paragraph(
                f'{area} has SOME ENVIRONMENTAL PROBLEMS that need attention. '
                f'The government should take steps to reduce pollution and plant more trees.',
                S(fontName='Helvetica', fontSize=10, textColor=BLACK, leading=16)))
        else:
            plain_lines.append(Paragraph(
                f'{area} is in GOOD environmental condition. The air is relatively clean '
                f'and temperatures are manageable. This should be maintained.',
                S(fontName='Helvetica', fontSize=10, textColor=BLACK, leading=16)))

        plain_lines.append(Spacer(1, 0.15*cm))
        plain_lines.append(Paragraph(
            f'The average temperature here is {avg_t} degrees Celsius — '
            f'{"very hot and dangerous" if avg_t > 35 else "above normal" if avg_t > 28 else "normal"}. '
            f'The air pollution (NO2) is {avg_n} micrograms per cubic meter — '
            f'{"dangerously high" if avg_n > 80 else "above safe limits" if avg_n > 40 else "within safe range"}. '
            f'Our computer found {a_cnt} days in the past year when the temperature was '
            f'dangerously unusual. '
            f'The next 30 days are expected to be around {f_avg} degrees, with temperature '
            f'{"going up" if trend == "Rising" else "going down" if trend == "Falling" else "staying about the same"}.',
            S(fontName='Helvetica', fontSize=10, textColor=DARK_GRAY, leading=16)))

        plain_box = Table([[p] for p in plain_lines], colWidths=[W - 0.6*cm])
        plain_box.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), TEAL_LIGHT),
            ('BOX',           (0,0), (-1,-1), 1.5, TEAL),
            ('TOPPADDING',    (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING',   (0,0), (-1,-1), 14),
            ('RIGHTPADDING',  (0,0), (-1,-1), 14),
        ]))
        cnt.append(plain_box)
        cnt.append(Spacer(1, 0.4*cm))

        # ══════════════════════
        #  SECTION 2 — TECHNICAL ANALYSIS
        # ══════════════════════
        cnt.append(HRFlowable(width=W, thickness=2, color=TEAL, spaceAfter=8))
        cnt.append(Paragraph('2. TECHNICAL ANALYSIS — ML MODEL RESULTS', s_section))
        cnt.append(Paragraph('2.1  Environmental Parameters', s_body_b))

        param_rows = [
            [Paragraph('Parameter', s_th),
             Paragraph('Measured Value', s_th),
             Paragraph('Status', s_th),
             Paragraph('Safe Limit', s_th)],
            [Paragraph('Land Surface Temperature (Average)\n(Simple: How hot the ground feels)', s_tc),
             Paragraph(f'{data.get("avg_temp","--")} degC', s_tc_b),
             Paragraph(tstat, S(fontName='Helvetica-Bold', fontSize=9,
                         textColor=tcol, leading=14)),
             Paragraph('Below 28 degC', s_tc)],
            [Paragraph('Land Surface Temperature (Maximum)\n(Simple: Hottest day recorded)', s_tc),
             Paragraph(f'{data.get("max_temp","--")} degC', s_tc_b),
             Paragraph('Recorded Peak', s_tc),
             Paragraph('Alert above 42 degC', s_tc)],
            [Paragraph('NO2 Concentration (Average)\n(Simple: How polluted the air is)', s_tc),
             Paragraph(f'{data.get("avg_no2","--")} ug/m3', s_tc_b),
             Paragraph(nstat, S(fontName='Helvetica-Bold', fontSize=9,
                         textColor=ncol, leading=14)),
             Paragraph('Below 40 ug/m3', s_tc)],
            [Paragraph('ARIMA 30-Day Forecast\n(Simple: What temperature next month)', s_tc),
             Paragraph(f'{data.get("forecast_avg","--")} degC', s_tc_b),
             Paragraph(f'Trend: {data.get("trend","Stable")}', s_tc),
             Paragraph('Next 30 days', s_tc)],
            [Paragraph('Anomaly Days Detected\n(Simple: Days with dangerous temperature)', s_tc),
             Paragraph(str(data.get('anomaly_count', 0)), s_tc_b),
             Paragraph('Detected by AI', s_tc),
             Paragraph('Ideal: 0 days', s_tc)],
            [Paragraph('Pollution Zones Found\n(Simple: How many danger zones on map)', s_tc),
             Paragraph(str(data.get('hotspot_clusters', 0)), s_tc_b),
             Paragraph('Mapped by AI', s_tc),
             Paragraph('Ideal: Less than 2', s_tc)],
            [Paragraph('Habitability Score\n(Simple: How good is this area to live in)', s_tc),
             Paragraph(f'{hs} out of 100', s_tc_b),
             Paragraph(hl, S(fontName='Helvetica-Bold', fontSize=9,
                         textColor=GREEN_GOOD if hs>60 else ORANGE if hs>40 else RED,
                         leading=14)),
             Paragraph('Good if above 60', s_tc)],
            [Paragraph('Overall Risk Score\n(Simple: Overall danger level)', s_tc),
             Paragraph(f'{rs} out of 100', s_tc_b),
             Paragraph(rl, S(fontName='Helvetica-Bold', fontSize=9,
                         textColor=rc(rs), leading=14)),
             Paragraph('Low if below 40', s_tc)],
        ]

        pt = Table(param_rows, colWidths=[W*0.33, W*0.19, W*0.22, W*0.26])
        pt.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0),  NAVY),
            ('TEXTCOLOR',     (0,0), (-1,0),  WHITE),
            ('BACKGROUND',    (0,1), (-1,-1), WHITE),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
            ('GRID',          (0,0), (-1,-1), 0.5, MID_GRAY),
            ('TOPPADDING',    (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING',   (0,0), (-1,-1), 8),
            ('RIGHTPADDING',  (0,0), (-1,-1), 8),
            ('VALIGN',        (0,0), (-1,-1), 'TOP'),
            ('LINEBELOW',     (0,0), (-1,0),  1.5, TEAL),
        ]))
        cnt.append(pt)
        cnt.append(Spacer(1, 0.4*cm))

        # What do these ML terms mean — explanation box
        explain_items = [
            Paragraph('<b>What do these technical terms mean?</b>',
                S(fontName='Helvetica-Bold', fontSize=9, textColor=TEAL, leading=13)),
            Paragraph(
                '<b>Isolation Forest:</b> A computer program that looks at 365 days of '
                'temperature data and automatically finds the days that were dangerously '
                'unusual — like extreme heat waves or sudden cold spells.',
                S(fontName='Helvetica', fontSize=9, textColor=DARK_GRAY,
                  leading=13, spaceAfter=4)),
            Paragraph(
                '<b>ARIMA Forecast:</b> A mathematical model that learns the seasonal '
                'temperature pattern of this area and predicts what the next 30 days '
                'will look like. Like a weather forecast but based on 1 year of data.',
                S(fontName='Helvetica', fontSize=9, textColor=DARK_GRAY,
                  leading=13, spaceAfter=4)),
            Paragraph(
                '<b>DBSCAN Clustering:</b> A program that groups 80 pollution measurement '
                'points on the map into zones — Red Alert, High Risk, Moderate, Safe. '
                'It helps identify exactly which parts of the area need urgent action.',
                S(fontName='Helvetica', fontSize=9, textColor=DARK_GRAY,
                  leading=13, spaceAfter=4)),
            Paragraph(
                '<b>NO2 (Nitrogen Dioxide):</b> A harmful gas produced mainly by vehicles '
                'and factories. Above 40 ug/m3 is unsafe. Above 80 ug/m3 is dangerous '
                'and can cause breathing problems, especially for children.',
                S(fontName='Helvetica', fontSize=9, textColor=DARK_GRAY,
                  leading=13, spaceAfter=0)),
        ]
        explain_box = Table([[p] for p in explain_items], colWidths=[W - 0.6*cm])
        explain_box.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), GOLD_LIGHT),
            ('BOX',           (0,0), (-1,-1), 0.5, GOLD),
            ('TOPPADDING',    (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING',   (0,0), (-1,-1), 12),
            ('RIGHTPADDING',  (0,0), (-1,-1), 12),
        ]))
        cnt.append(explain_box)
        cnt.append(Spacer(1, 0.4*cm))

        # ── 2.2 Anomaly table ──
        anomalies = data.get('anomalies', [])
        if anomalies:
            cnt.append(Paragraph(
                f'2.2  Unusual Temperature Days  ({len(anomalies)} dangerous days found)',
                s_body_b))
            cnt.append(Paragraph(
                'These are the days when temperature was dangerously different from normal.',
                s_note))

            ar = [[
                Paragraph('Date', s_th),
                Paragraph('Temperature', s_th),
                Paragraph('How different from normal', s_th),
                Paragraph('How serious', s_th),
            ]]
            avgt = data.get('avg_temp', 30)
            for a in anomalies[:15]:
                dev = round(a.get('temp', avgt) - avgt, 1)
                ds  = (f'+{dev} degC hotter than normal'
                       if dev > 0 else f'{abs(dev)} degC colder than normal')
                sev = a.get('severity', 'Low')
                sc  = RED if sev=='High' else ORANGE if sev=='Medium' else GREEN_GOOD
                sl  = ('Very Unusual (serious health risk)'
                       if sev=='High' else
                       'Unusual (be careful outdoors)'
                       if sev=='Medium' else
                       'Slightly Unusual')
                ar.append([
                    Paragraph(a.get('date', '--'), s_tc),
                    Paragraph(f'{a.get("temp","--")} degC', s_tc_b),
                    Paragraph(ds, S(fontName='Helvetica', fontSize=9,
                        textColor=RED if dev > 0 else TEAL, leading=13)),
                    Paragraph(sl, S(fontName='Helvetica-Bold', fontSize=9,
                        textColor=sc, leading=13)),
                ])

            at = Table(ar, colWidths=[W*0.18, W*0.17, W*0.35, W*0.30])
            at.setStyle(TableStyle([
                ('BACKGROUND',    (0,0), (-1,0),  NAVY),
                ('BACKGROUND',    (0,1), (-1,-1), WHITE),
                ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
                ('GRID',          (0,0), (-1,-1), 0.5, MID_GRAY),
                ('TOPPADDING',    (0,0), (-1,-1), 7),
                ('BOTTOMPADDING', (0,0), (-1,-1), 7),
                ('LEFTPADDING',   (0,0), (-1,-1), 8),
                ('RIGHTPADDING',  (0,0), (-1,-1), 8),
                ('VALIGN',        (0,0), (-1,-1), 'TOP'),
                ('LINEBELOW',     (0,0), (-1,0),  1.5, TEAL),
            ]))
            cnt.append(at)
            cnt.append(Spacer(1, 0.4*cm))

        # ══════════════════════
        #  SECTION 3 — AI ACTION PLAN
        # ══════════════════════
        cnt.append(HRFlowable(width=W, thickness=2, color=TEAL, spaceAfter=8))
        cnt.append(Paragraph('3. AI-GENERATED ACTION PLAN  (What should be done)', s_section))
        cnt.append(Paragraph(
            'The following recommendations were generated by Claude AI (Anthropic) '
            'based on all the satellite and ML data above. These are specific, '
            'practical steps for this area.',
            s_body))
        cnt.append(Spacer(1, 0.2*cm))

        try:
            advice_text = get_ai_advice(data)
        except Exception:
            advice_text = "AI advice could not be generated. Please check your Anthropic API key."

        adv_paras = []
        for line in advice_text.strip().split('\n'):
            line = line.strip()
            if not line:
                adv_paras.append(Spacer(1, 0.1*cm))
                continue
            if re.match(r'^\d\.', line):
                adv_paras.append(Paragraph(line, s_adv_h))
            elif line.startswith('-') or line.startswith('*'):
                adv_paras.append(Paragraph(
                    '  -  ' + line.lstrip('-*').strip(), s_adv_b))
            elif line.startswith('Week'):
                adv_paras.append(Paragraph(
                    '<b>' + line + '</b>', s_adv_b))
            else:
                adv_paras.append(Paragraph(line, s_adv_b))

        if adv_paras:
            adv_rows = [[p] for p in adv_paras]
            adv_tbl  = Table(adv_rows, colWidths=[W - 0.8*cm])
            adv_tbl.setStyle(TableStyle([
                ('BACKGROUND',    (0,0), (-1,-1), TEAL_LIGHT),
                ('BOX',           (0,0), (-1,-1), 1.5, TEAL),
                ('TOPPADDING',    (0,0), (-1,-1), 3),
                ('BOTTOMPADDING', (0,0), (-1,-1), 3),
                ('LEFTPADDING',   (0,0), (-1,-1), 16),
                ('RIGHTPADDING',  (0,0), (-1,-1), 12),
            ]))
            cnt.append(adv_tbl)
        cnt.append(Spacer(1, 0.5*cm))

        # ══════════════════════
        #  SECTION 4 — METHODOLOGY
        # ══════════════════════
        cnt.append(HRFlowable(width=W, thickness=2, color=TEAL, spaceAfter=8))
        cnt.append(Paragraph('4. HOW THIS REPORT WAS MADE  (Methodology)', s_section))

        meth_rows = [
            [Paragraph('Method', s_th),
             Paragraph('What it does', s_th),
             Paragraph('Simple explanation', s_th),
             Paragraph('Data source', s_th)],
            [Paragraph('Isolation Forest', s_tc_b),
             Paragraph('Anomaly detection', s_tc),
             Paragraph('Finds dangerous unusual days', s_tc),
             Paragraph('NASA POWER / MODIS', s_tc)],
            [Paragraph('ARIMA (3,1,0)', s_tc_b),
             Paragraph('Temperature forecast', s_tc),
             Paragraph('Predicts next 30 days', s_tc),
             Paragraph('NASA POWER / MODIS', s_tc)],
            [Paragraph('DBSCAN Clustering', s_tc_b),
             Paragraph('Pollution zone mapping', s_tc),
             Paragraph('Groups pollution into danger zones', s_tc),
             Paragraph('OWM / Sentinel-5P / WAQI', s_tc)],
        ]
        mt = Table(meth_rows, colWidths=[W*0.22, W*0.22, W*0.30, W*0.26])
        mt.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0),  NAVY),
            ('BACKGROUND',    (0,1), (-1,-1), WHITE),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
            ('GRID',          (0,0), (-1,-1), 0.5, MID_GRAY),
            ('TOPPADDING',    (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('LEFTPADDING',   (0,0), (-1,-1), 8),
            ('RIGHTPADDING',  (0,0), (-1,-1), 8),
            ('VALIGN',        (0,0), (-1,-1), 'TOP'),
            ('LINEBELOW',     (0,0), (-1,0),  1.5, TEAL),
        ]))
        cnt.append(mt)
        cnt.append(Spacer(1, 0.5*cm))

        # ══════════════════════
        #  DISCLAIMER
        # ══════════════════════
        cnt.append(HRFlowable(width=W, thickness=0.5, color=MID_GRAY, spaceAfter=8))
        disc = Table([[Paragraph(
            '<b>Disclaimer:</b> This report is produced by GreenGrid using satellite data from '
            'NASA POWER, MODIS, and OpenWeatherMap. Data labelled "Simulated" uses '
            'statistically realistic models based on known climate patterns for Gujarat. '
            'For official regulatory decisions, cross-verify with CPCB and GPCB data.',
            S(fontName='Helvetica', fontSize=8, textColor=DARK_GRAY,
              leading=13, alignment=TA_JUSTIFY)
        )]], colWidths=[W])
        disc.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,-1), GOLD_LIGHT),
            ('BOX',           (0,0), (-1,-1), 0.5, GOLD),
            ('TOPPADDING',    (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 10),
            ('LEFTPADDING',   (0,0), (-1,-1), 12),
            ('RIGHTPADDING',  (0,0), (-1,-1), 12),
        ]))
        cnt.append(disc)
        cnt.append(Spacer(1, 0.4*cm))

        # ══════════════════════
        #  SIGNATURE + FOOTER
        # ══════════════════════
        sig = Table([[
            Paragraph(
                f'<b>Prepared by:</b> GreenGrid AI Platform<br/>'
                f'<b>Date:</b> {today}<br/>'
                f'<b>Reference:</b> {ref_no}',
                S(fontName='Helvetica', fontSize=9, textColor=DARK_GRAY, leading=14)),
            Paragraph(
                f'<b>Area:</b> {area}, {city}<br/>'
                f'<b>Classification:</b> {data.get("type","--").title()}<br/>'
                f'<b>Risk Level:</b> {rl}',
                S(fontName='Helvetica', fontSize=9, textColor=DARK_GRAY,
                  leading=14, alignment=TA_RIGHT)),
        ]], colWidths=[W*0.5, W*0.5])
        sig.setStyle(TableStyle([
            ('LINEABOVE',     (0,0), (-1,0), 1, MID_GRAY),
            ('TOPPADDING',    (0,0), (-1,-1), 8),
            ('BOTTOMPADDING', (0,0), (-1,-1), 8),
            ('VALIGN',        (0,0), (-1,-1), 'TOP'),
        ]))
        cnt.append(sig)

        footer = Table([[Paragraph(
            f'GreenGrid Environmental Intelligence Platform  |  {today}  |  '
            f'Ref: {ref_no}  |  Data: {data_period}',
            s_footer
        )]], colWidths=[W])
        footer.setStyle(TableStyle([
            ('LINEABOVE',     (0,0), (-1,-1), 0.5, MID_GRAY),
            ('BACKGROUND',    (0,0), (-1,-1), LIGHT_GRAY),
            ('TOPPADDING',    (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        cnt.append(footer)

        doc.build(cnt)
        buf.seek(0)

        fname = (
            f'GreenGrid_{city}_{area.replace(" ","_")}'
            f'_Report_{datetime.now().strftime("%Y%m%d")}.pdf'
        )
        return send_file(buf, as_attachment=True,
                         download_name=fname, mimetype='application/pdf')

    except ImportError:
        return jsonify({'error': 'Run: pip install reportlab'}), 500
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  EcoSentinel + GreenGrid Server Starting...")
    print("  Home (GreenGrid map): http://localhost:5000")
    print("  ML Dashboard:         http://localhost:5000/dashboard")
    print("  API Health check:     http://localhost:5000/api/health")
    print("  Manual refresh:       http://localhost:5000/api/refresh")
    print("=" * 60)
    app.run(debug=True, port=5000)
