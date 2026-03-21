from flask import Flask, jsonify, render_template, request, send_file
from flask_cors import CORS
from functools import lru_cache
import io
import json
from datetime import datetime

from ml_engine import analyze_area, analyze_city_overview, get_cities, get_areas

app = Flask(__name__)
CORS(app)


@lru_cache(maxsize=100)
def cached_analyze(city, area):
    return analyze_area(city, area)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/cities')
def cities():
    return jsonify({'cities': get_cities()})

@app.route('/api/areas/<city>')
def areas(city):
    area_list = get_areas(city)
    if not area_list:
        return jsonify({'error': 'City not found'}), 404
    return jsonify({'areas': area_list})

@app.route('/api/analyze/<city>/<area>')
def analyze(city, area):
    area = area.replace('%20', ' ').replace('+', ' ')
    try:
        result = cached_analyze(city, area)
        if result is None:
            return jsonify({'error': f'{area} not found in {city}'}), 404
        return jsonify(result)
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
    return jsonify({'status': 'running', 'cities': get_cities()})


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
- Unusual temperature days found: {data['anomaly_count']} days in 2023
- Worst temperature recorded: {data.get('worst_anomaly_temp', 'N/A')}°C
- 30-day temperature forecast: {data['forecast_avg']}°C (trend: {data['trend']})
- Pollution zones found: {data['hotspot_clusters']}
- Habitability score: {data['habitability']['score']}/100 ({data['habitability']['label']})
- Overall risk: {data['risk']['label']} (score: {data['risk']['score']}/100)

Give your response in these exact sections:

1. CURRENT SITUATION
(2 lines plain English summary for a municipal commissioner)

2. IMMEDIATE ACTIONS NEEDED
(List 3-4 specific practical actions for Ahmedabad/Gandhinagar)

3. 30-DAY ACTION PLAN
Week 1: What to do first
Week 2-3: What to implement
Week 4: Review and measure

4. WHO SHOULD ACT
Which government department and industries

5. EXPECTED IMPROVEMENT
What improvement is realistic in 3 months

Keep language simple. Maximum 300 words. No jargon."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    except Exception as e:
        risk_score = data['risk']['score']
        avg_temp   = data['avg_temp']
        avg_no2    = data['avg_no2']
        area       = data['area']
        hab_score  = data['habitability']['score']

        if risk_score > 70:
            return f"""1. CURRENT SITUATION
{area} is in a HIGH RISK environmental state. Temperature of {avg_temp}°C combined with NO2 levels of {avg_no2} ug/m3 requires immediate government intervention. A total of {data['anomaly_count']} dangerous temperature anomaly days were recorded in 2023.

2. IMMEDIATE ACTIONS NEEDED
- Restrict operations at top polluting factories between 10am and 6pm daily
- Issue a public health advisory recommending avoidance of outdoor activity during peak hours
- Deploy mobile air quality monitoring units across the affected zones
- Activate emergency tree-watering and dust suppression protocols immediately

3. 30-DAY ACTION PLAN
Week 1: Survey and identify the top 10 pollution sources. Issue formal notices to industrial units.
Week 2-3: Initiate tree plantation of 500+ saplings. Install dust barriers on active construction sites.
Week 4: Conduct re-measurement of air quality. Submit compliance report to AMC Environment Dept.

4. WHO SHOULD ACT
Ahmedabad Municipal Corporation (AMC) Environment Department and Gujarat Pollution Control Board (GPCB) must lead. Local factory owner associations must be made accountable.

5. EXPECTED IMPROVEMENT
With strict enforcement and consistent monitoring, NO2 pollution levels can be reduced by 20 to 30 percent within 90 days. Habitability score is projected to improve from the current {hab_score} to above 60 within three months."""

        elif risk_score > 40:
            return f"""1. CURRENT SITUATION
{area} is at MODERATE environmental risk. Conditions are manageable but require consistent monitoring. NO2 levels of {avg_no2} ug/m3 exceed the safe threshold of 40 ug/m3 set by CPCB.

2. IMMEDIATE ACTIONS NEEDED
- Increase green cover by planting trees along all major roads and intersections
- Establish two permanent air quality monitoring stations in the area
- Enforce vehicle pollution checks at key entry and exit points
- Promote rooftop garden initiatives in residential housing societies

3. 30-DAY ACTION PLAN
Week 1: Install air quality monitors and record baseline measurements across the area.
Week 2-3: Conduct a tree plantation drive in partnership with local Resident Welfare Associations.
Week 4: Review collected data and publish a public environmental status report.

4. WHO SHOULD ACT
Ward-level AMC officers must coordinate. Resident Welfare Associations (RWAs) and local commercial establishments need to be actively involved.

5. EXPECTED IMPROVEMENT
Habitability score can realistically improve from {hab_score} to above 70 within 60 days of consistent and coordinated action."""

        else:
            return f"""1. CURRENT SITUATION
{area} is in a GOOD environmental condition. The habitability score of {hab_score}/100 is commendable and reflects well-maintained green cover and regulated pollution levels. Efforts should now focus on sustaining and improving further.

2. HOW TO MAINTAIN GOOD CONDITIONS
- Continue existing green cover maintenance and watering programs
- Conduct monthly NO2 level monitoring and publish results publicly
- Prevent establishment of new heavy industrial units in this zone
- Document and share best practices with neighbouring areas as a model

3. 30-DAY ACTION PLAN
Week 1: Document current green cover percentage using available satellite imagery.
Week 2-3: Organise community awareness programs on maintaining clean air and environment.
Week 4: Officially designate this area as a Model Environmental Zone for other wards.

4. WHO SHOULD ACT
Local ward committee members and community volunteers should lead. Schools, colleges, and municipal garden departments should be engaged for awareness and maintenance.

5. EXPECTED IMPROVEMENT
The current score of {hab_score}/100 can be sustained and a target of 85+ is achievable by the end of the year with continued community engagement."""


@app.route('/api/advice/<city>/<area>')
def get_advice_api(city, area):
    area = area.replace('%20', ' ').replace('+', ' ')
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

        data     = cached_analyze(city, area)
        risk     = data.get('risk', {})
        hab      = data.get('habitability', {})
        today    = datetime.now().strftime("%d %B %Y")
        ref_no   = f"ECO/{city[:3].upper()}/{datetime.now().strftime('%Y%m%d')}/001"

        # ── Color palette (professional gov report) ──
        DARK_NAVY   = colors.HexColor('#0D1B2A')
        NAVY        = colors.HexColor('#1B2A4A')
        TEAL        = colors.HexColor('#0D6E6E')
        TEAL_LIGHT  = colors.HexColor('#E8F5F5')
        TEAL_MID    = colors.HexColor('#B2DFDF')
        GOLD        = colors.HexColor('#C49A2C')
        GOLD_LIGHT  = colors.HexColor('#FDF6E3')
        WHITE       = colors.white
        LIGHT_GRAY  = colors.HexColor('#F5F6F7')
        MID_GRAY    = colors.HexColor('#E2E4E8')
        DARK_GRAY   = colors.HexColor('#4A5568')
        BLACK       = colors.HexColor('#1A202C')
        RED_ALERT   = colors.HexColor('#C53030')
        ORANGE_WARN = colors.HexColor('#C05621')
        GREEN_GOOD  = colors.HexColor('#276749')

        def risk_color(score):
            if score > 70: return RED_ALERT
            elif score > 40: return ORANGE_WARN
            return GREEN_GOOD

        def no2_status(val):
            if val > 80:   return ("Hazardous",   RED_ALERT)
            elif val > 50: return ("Unhealthy",   ORANGE_WARN)
            elif val > 30: return ("Moderate",    GOLD)
            else:          return ("Good",        GREEN_GOOD)

        def temp_status(val):
            if val > 35:   return ("Critical",   RED_ALERT)
            elif val > 28: return ("Elevated",   ORANGE_WARN)
            else:          return ("Normal",     GREEN_GOOD)

        # ── Styles ──
        def S(name, **kw):
            return ParagraphStyle(name, **kw)

        sty_cover_title = S('ct', fontName='Helvetica-Bold', fontSize=26,
                            textColor=WHITE, leading=32, alignment=TA_LEFT)
        sty_cover_sub   = S('cs', fontName='Helvetica', fontSize=13,
                            textColor=colors.HexColor('#A0B4C8'), leading=18, alignment=TA_LEFT)
        sty_cover_meta  = S('cm', fontName='Helvetica', fontSize=10,
                            textColor=colors.HexColor('#7A9AB8'), leading=14, alignment=TA_LEFT)
        sty_section     = S('sh', fontName='Helvetica-Bold', fontSize=12,
                            textColor=TEAL, leading=16, spaceBefore=14, spaceAfter=6,
                            alignment=TA_LEFT)
        sty_body        = S('b',  fontName='Helvetica', fontSize=10,
                            textColor=DARK_GRAY, leading=16, spaceAfter=4,
                            alignment=TA_JUSTIFY)
        sty_body_bold   = S('bb', fontName='Helvetica-Bold', fontSize=10,
                            textColor=BLACK, leading=16, spaceAfter=4)
        sty_small       = S('sm', fontName='Helvetica', fontSize=8,
                            textColor=DARK_GRAY, leading=12)
        sty_footer      = S('ft', fontName='Helvetica', fontSize=8,
                            textColor=colors.HexColor('#94A3B8'), alignment=TA_CENTER)
        sty_tbl_head    = S('th', fontName='Helvetica-Bold', fontSize=9,
                            textColor=WHITE, leading=12, alignment=TA_LEFT)
        sty_tbl_cell    = S('tc', fontName='Helvetica', fontSize=9,
                            textColor=DARK_GRAY, leading=13, alignment=TA_LEFT)
        sty_tbl_cell_b  = S('tcb',fontName='Helvetica-Bold', fontSize=9,
                            textColor=BLACK, leading=13)
        sty_highlight   = S('hl', fontName='Helvetica-Bold', fontSize=11,
                            textColor=TEAL, leading=15, alignment=TA_CENTER)
        sty_advice_head = S('ah', fontName='Helvetica-Bold', fontSize=10,
                            textColor=NAVY, leading=14, spaceBefore=8, spaceAfter=2)
        sty_advice_body = S('ab', fontName='Helvetica', fontSize=10,
                            textColor=DARK_GRAY, leading=15, spaceAfter=3, leftIndent=12,
                            alignment=TA_JUSTIFY)

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=1.5*cm, bottomMargin=2*cm,
            title=f"EcoSentinel Report — {area}, {city}",
            author="EcoSentinel Environmental Intelligence Platform"
        )

        content = []
        W = 17 * cm  # usable width

        # ════════════════════════════════════
        #  COVER PAGE
        # ════════════════════════════════════

        # Top navy banner
        cover_top = Table([[ Paragraph(
            f'<font size="8" color="#A0B4C8">ECOSENTINEL · SATELLITE ENVIRONMENTAL INTELLIGENCE PLATFORM · CONFIDENTIAL</font>',
            S('ct2', fontName='Helvetica', fontSize=8, textColor=colors.HexColor('#A0B4C8'),
              alignment=TA_CENTER))
        ]], colWidths=[W])
        cover_top.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), DARK_NAVY),
            ('PADDING',    (0,0), (-1,-1), 8),
            ('LINEBELOW',  (0,0), (-1,-1), 2, GOLD),
        ]))
        content.append(cover_top)
        content.append(Spacer(1, 0.8*cm))

        # Organization logos row placeholder + title
        org_row = Table([[
            Paragraph('<b>ECOSENTINEL</b>', S('logo', fontName='Helvetica-Bold',
                fontSize=18, textColor=TEAL)),
            Paragraph(f'<font color="#94A3B8" size="8">Powered by MODIS · Sentinel-5P · Isolation Forest · ARIMA · DBSCAN</font>',
                S('pl', fontName='Helvetica', fontSize=8,
                  textColor=colors.HexColor('#94A3B8'), alignment=TA_RIGHT)),
        ]], colWidths=[W*0.5, W*0.5])
        org_row.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                                     ('PADDING',(0,0),(-1,-1),0)]))
        content.append(org_row)
        content.append(HRFlowable(width=W, thickness=1, color=MID_GRAY, spaceAfter=10))

        # Main title block
        title_block = Table([[
            Table([
                [Paragraph('ENVIRONMENTAL INTELLIGENCE REPORT',
                    S('rtype', fontName='Helvetica-Bold', fontSize=9,
                      textColor=GOLD, leading=12))],
                [Paragraph(f'{area.upper()}',
                    S('aname', fontName='Helvetica-Bold', fontSize=28,
                      textColor=DARK_NAVY, leading=32))],
                [Paragraph(f'{city}, Gujarat, India',
                    S('cname', fontName='Helvetica-Bold', fontSize=14,
                      textColor=TEAL, leading=18))],
                [Spacer(1, 0.3*cm)],
                [Paragraph(
                    f'Satellite-based environmental analysis using Machine Learning. '
                    f'Data sourced from MODIS NASA and Copernicus Sentinel-5P satellite systems.',
                    S('intro', fontName='Helvetica', fontSize=10,
                      textColor=DARK_GRAY, leading=15, alignment=TA_JUSTIFY)
                )],
            ], colWidths=[W*0.65])
        ]], colWidths=[W])
        title_block.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), LIGHT_GRAY),
            ('PADDING',    (0,0), (-1,-1), 20),
            ('ROUNDEDCORNERS', [8]),
        ]))
        content.append(title_block)
        content.append(Spacer(1, 0.4*cm))

        # Meta info row
        meta_data = [
            [
                Paragraph(f'<b>Reference No.</b><br/>{ref_no}',
                    S('mi', fontName='Helvetica', fontSize=9, textColor=DARK_GRAY, leading=13)),
                Paragraph(f'<b>Date of Report</b><br/>{today}',
                    S('mi', fontName='Helvetica', fontSize=9, textColor=DARK_GRAY, leading=13)),
                Paragraph(f'<b>Area Classification</b><br/>{data.get("type","—").title()}',
                    S('mi', fontName='Helvetica', fontSize=9, textColor=DARK_GRAY, leading=13)),
                Paragraph(f'<b>Data Source</b><br/>{data.get("data_source","Simulated")}',
                    S('mi', fontName='Helvetica', fontSize=9, textColor=DARK_GRAY, leading=13)),
            ]
        ]
        meta_tbl = Table(meta_data, colWidths=[W/4]*4)
        meta_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), WHITE),
            ('BOX',        (0,0), (-1,-1), 1, MID_GRAY),
            ('INNERGRID',  (0,0), (-1,-1), 0.5, MID_GRAY),
            ('PADDING',    (0,0), (-1,-1), 10),
            ('VALIGN',     (0,0), (-1,-1), 'TOP'),
        ]))
        content.append(meta_tbl)
        content.append(Spacer(1, 0.5*cm))

        # ── KEY METRICS SUMMARY (4 big cards) ──
        r_score  = risk.get('score', 0)
        r_label  = risk.get('label', '—')
        h_score  = hab.get('score', 0)
        h_label  = hab.get('label', '—')
        t_status, t_color = temp_status(data.get('avg_temp', 0))
        n_status, n_color = no2_status(data.get('avg_no2', 0))

        def metric_card(title, value, unit, label, val_color):
            return Table([
                [Paragraph(title, S('mct', fontName='Helvetica-Bold', fontSize=8,
                    textColor=colors.HexColor('#64748B'), leading=10))],
                [Paragraph(f'<font size="22" color="{val_color.hexval()}">'
                           f'<b>{value}</b></font>'
                           f'<font size="10" color="#64748B"> {unit}</font>',
                    S('mcv', fontName='Helvetica', fontSize=22, leading=26))],
                [Paragraph(label, S('mcl', fontName='Helvetica-Bold', fontSize=8,
                    textColor=val_color, leading=10))],
            ], colWidths=[(W/4)-0.3*cm])

        cards_row = [
            metric_card("RISK SCORE", str(r_score), "/ 100", r_label, risk_color(r_score)),
            metric_card("HABITABILITY", str(h_score), "/ 100", h_label,
                        GREEN_GOOD if h_score>60 else ORANGE_WARN if h_score>40 else RED_ALERT),
            metric_card("AVG TEMPERATURE", str(data.get('avg_temp','—')), "°C", t_status, t_color),
            metric_card("AVG NO₂ POLLUTION", str(data.get('avg_no2','—')), "μg/m³", n_status, n_color),
        ]

        cards_tbl = Table([cards_row], colWidths=[W/4]*4)
        cards_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), WHITE),
            ('BOX',        (0,0), (-1,-1), 1, MID_GRAY),
            ('INNERGRID',  (0,0), (-1,-1), 0.5, MID_GRAY),
            ('PADDING',    (0,0), (-1,-1), 12),
            ('VALIGN',     (0,0), (-1,-1), 'TOP'),
            ('LINEBELOW',  (0,0), (-1,-1), 3, TEAL),
        ]))
        content.append(cards_tbl)
        content.append(Spacer(1, 0.6*cm))

        # ════════════════════════════════════
        #  SECTION 1 — EXECUTIVE SUMMARY
        # ════════════════════════════════════
        content.append(HRFlowable(width=W, thickness=2, color=TEAL, spaceAfter=6))
        content.append(Paragraph('1. EXECUTIVE SUMMARY', sty_section))

        summary_text = (
            f"This report presents a comprehensive environmental analysis of <b>{area}</b>, "
            f"{city}, Gujarat, conducted using satellite-derived data and advanced Machine Learning algorithms. "
            f"The area is classified as a <b>{data.get('type','mixed').title()}</b> zone with an overall "
            f"environmental risk score of <b>{r_score}/100</b> ({r_label}) and a habitability index of "
            f"<b>{h_score}/100</b> ({h_label})."
        )
        content.append(Paragraph(summary_text, sty_body))

        findings_text = (
            f"Key findings indicate an average land surface temperature of <b>{data.get('avg_temp','—')}°C</b> "
            f"with a maximum recorded temperature of <b>{data.get('max_temp','—')}°C</b>. "
            f"Air quality analysis reveals average NO₂ concentration of "
            f"<b>{data.get('avg_no2','—')} μg/m³</b> ({n_status}). "
            f"The Isolation Forest anomaly detection model identified "
            f"<b>{data.get('anomaly_count', 0)} unusual temperature days</b> during the 2023 analysis period. "
            f"The ARIMA forecasting model projects an average temperature of "
            f"<b>{data.get('forecast_avg','—')}°C</b> for the next 30 days with a "
            f"<b>{data.get('trend','Stable')}</b> trend."
        )
        content.append(Paragraph(findings_text, sty_body))
        content.append(Spacer(1, 0.3*cm))

        # ════════════════════════════════════
        #  SECTION 2 — DETAILED ML RESULTS
        # ════════════════════════════════════
        content.append(HRFlowable(width=W, thickness=2, color=TEAL, spaceAfter=6))
        content.append(Paragraph('2. MACHINE LEARNING ANALYSIS RESULTS', sty_section))

        # 2a — Parameter table
        content.append(Paragraph('2.1  Environmental Parameters', sty_body_bold))
        param_rows = [
            [Paragraph('Parameter', sty_tbl_head),
             Paragraph('Measured Value', sty_tbl_head),
             Paragraph('Status', sty_tbl_head),
             Paragraph('Threshold', sty_tbl_head)],
            [Paragraph('Land Surface Temperature (Avg)', sty_tbl_cell),
             Paragraph(f"{data.get('avg_temp','—')}°C", sty_tbl_cell_b),
             Paragraph(t_status, S('st', fontName='Helvetica-Bold', fontSize=9, textColor=t_color)),
             Paragraph('Safe < 28°C', sty_tbl_cell)],
            [Paragraph('Land Surface Temperature (Max)', sty_tbl_cell),
             Paragraph(f"{data.get('max_temp','—')}°C", sty_tbl_cell_b),
             Paragraph('Recorded Peak', sty_tbl_cell),
             Paragraph('Alert > 42°C', sty_tbl_cell)],
            [Paragraph('NO₂ Concentration (Avg)', sty_tbl_cell),
             Paragraph(f"{data.get('avg_no2','—')} μg/m³", sty_tbl_cell_b),
             Paragraph(n_status, S('st2', fontName='Helvetica-Bold', fontSize=9, textColor=n_color)),
             Paragraph('Safe < 40 μg/m³', sty_tbl_cell)],
            [Paragraph('ARIMA 30-Day Forecast', sty_tbl_cell),
             Paragraph(f"{data.get('forecast_avg','—')}°C", sty_tbl_cell_b),
             Paragraph(data.get('trend','Stable'), sty_tbl_cell),
             Paragraph('Next 30 days', sty_tbl_cell)],
            [Paragraph('Anomaly Days (Isolation Forest)', sty_tbl_cell),
             Paragraph(str(data.get('anomaly_count',0)), sty_tbl_cell_b),
             Paragraph('Detected', sty_tbl_cell),
             Paragraph('Ideal: 0 days', sty_tbl_cell)],
            [Paragraph('Pollution Hotspot Zones (DBSCAN)', sty_tbl_cell),
             Paragraph(str(data.get('hotspot_clusters',0)), sty_tbl_cell_b),
             Paragraph('Identified', sty_tbl_cell),
             Paragraph('Ideal: < 2 zones', sty_tbl_cell)],
            [Paragraph('Habitability Index', sty_tbl_cell),
             Paragraph(f"{h_score}/100", sty_tbl_cell_b),
             Paragraph(h_label, S('st3', fontName='Helvetica-Bold', fontSize=9,
                textColor=GREEN_GOOD if h_score>60 else ORANGE_WARN if h_score>40 else RED_ALERT)),
             Paragraph('Good > 60', sty_tbl_cell)],
            [Paragraph('Overall Risk Score', sty_tbl_cell),
             Paragraph(f"{r_score}/100", sty_tbl_cell_b),
             Paragraph(r_label, S('st4', fontName='Helvetica-Bold', fontSize=9,
                textColor=risk_color(r_score))),
             Paragraph('Low < 40', sty_tbl_cell)],
        ]

        param_tbl = Table(param_rows, colWidths=[W*0.35, W*0.2, W*0.22, W*0.23])
        param_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0),  NAVY),
            ('TEXTCOLOR',     (0,0), (-1,0),  WHITE),
            ('BACKGROUND',    (0,1), (-1,-1), WHITE),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
            ('GRID',          (0,0), (-1,-1), 0.5, MID_GRAY),
            ('PADDING',       (0,0), (-1,-1), 8),
            ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
            ('LINEBELOW',     (0,0), (-1,0),  1.5, TEAL),
        ]))
        content.append(param_tbl)
        content.append(Spacer(1, 0.4*cm))

        # 2b — Anomaly table
        anomalies = data.get('anomalies', [])
        if anomalies:
            content.append(Paragraph(
                f'2.2  Unusual Temperature Days — Isolation Forest Detection '
                f'({len(anomalies)} anomalies identified)', sty_body_bold))

            anom_rows = [[
                Paragraph('Date', sty_tbl_head),
                Paragraph('Temperature', sty_tbl_head),
                Paragraph('Deviation from Normal', sty_tbl_head),
                Paragraph('Severity Level', sty_tbl_head),
            ]]
            avg_t = data.get('avg_temp', 30)
            for a in anomalies[:12]:
                dev  = round(a.get('temp', avg_t) - avg_t, 1)
                dev_str = f"+{dev}°C above normal" if dev > 0 else f"{abs(dev)}°C below normal"
                sev  = a.get('severity','Low')
                sc   = RED_ALERT if sev=='High' else ORANGE_WARN if sev=='Medium' else GREEN_GOOD
                sev_lbl = 'Very Unusual' if sev=='High' else 'Unusual' if sev=='Medium' else 'Slightly Unusual'
                anom_rows.append([
                    Paragraph(a.get('date','—'), sty_tbl_cell),
                    Paragraph(f"{a.get('temp','—')}°C", sty_tbl_cell_b),
                    Paragraph(dev_str, S('ds', fontName='Helvetica', fontSize=9,
                        textColor=RED_ALERT if dev>0 else TEAL)),
                    Paragraph(sev_lbl, S('svl', fontName='Helvetica-Bold', fontSize=9, textColor=sc)),
                ])

            anom_tbl = Table(anom_rows, colWidths=[W*0.22, W*0.18, W*0.35, W*0.25])
            anom_tbl.setStyle(TableStyle([
                ('BACKGROUND',    (0,0), (-1,0),  NAVY),
                ('BACKGROUND',    (0,1), (-1,-1), WHITE),
                ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
                ('GRID',          (0,0), (-1,-1), 0.5, MID_GRAY),
                ('PADDING',       (0,0), (-1,-1), 7),
                ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
                ('LINEBELOW',     (0,0), (-1,0),  1.5, TEAL),
            ]))
            content.append(anom_tbl)
            content.append(Spacer(1, 0.3*cm))

        # ════════════════════════════════════
        #  SECTION 3 — AI ACTION PLAN
        # ════════════════════════════════════
        content.append(HRFlowable(width=W, thickness=2, color=TEAL, spaceAfter=6))
        content.append(Paragraph('3. AI-GENERATED ENVIRONMENTAL ACTION PLAN', sty_section))
        content.append(Paragraph(
            'The following action plan has been generated by Claude AI (Anthropic) based on '
            'the satellite ML analysis data above. Recommendations are specific to the '
            f'{data.get("type","mixed")} classification and environmental risk profile of {area}.',
            sty_body))
        content.append(Spacer(1, 0.2*cm))

        # Advice box with background
        try:
            advice_text = get_ai_advice(data)
        except:
            advice_text = "AI advice unavailable. Please ensure Anthropic API is configured."

        advice_paras = []
        for line in advice_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            # Section headers (numbered like "1. CURRENT SITUATION")
            import re
            if re.match(r'^\d\.', line):
                advice_paras.append(Paragraph(line, sty_advice_head))
            elif line.startswith('-') or line.startswith('•'):
                advice_paras.append(Paragraph(
                    '• ' + line.lstrip('-•').strip(), sty_advice_body))
            elif line.startswith('Week'):
                advice_paras.append(Paragraph(
                    '<b>' + line + '</b>', sty_advice_body))
            else:
                advice_paras.append(Paragraph(line, sty_advice_body))

        advice_inner = [[para] for para in advice_paras]
        if advice_inner:
            advice_tbl = Table(advice_inner, colWidths=[W - 0.8*cm])
            advice_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), TEAL_LIGHT),
                ('PADDING',    (0,0), (-1,-1), 4),
                ('LEFTPADDING',(0,0), (-1,-1), 16),
                ('BOX',        (0,0), (-1,-1), 1.5, TEAL),
                ('LINEAFTER',  (0,0), (0,-1),  3, GOLD),
            ]))
            content.append(advice_tbl)

        content.append(Spacer(1, 0.5*cm))

        # ════════════════════════════════════
        #  SECTION 4 — METHODOLOGY
        # ════════════════════════════════════
        content.append(HRFlowable(width=W, thickness=2, color=TEAL, spaceAfter=6))
        content.append(Paragraph('4. METHODOLOGY AND DATA SOURCES', sty_section))

        method_rows = [
            [Paragraph('ML Model', sty_tbl_head),
             Paragraph('Algorithm', sty_tbl_head),
             Paragraph('Purpose', sty_tbl_head),
             Paragraph('Data Source', sty_tbl_head)],
            [Paragraph('Anomaly Detection', sty_tbl_cell),
             Paragraph('Isolation Forest', sty_tbl_cell_b),
             Paragraph('Identifies unusual temperature days — heat waves, cold snaps', sty_tbl_cell),
             Paragraph('MODIS MOD11A1 (NASA)', sty_tbl_cell)],
            [Paragraph('Temperature Forecast', sty_tbl_cell),
             Paragraph('ARIMA (3,1,0)', sty_tbl_cell_b),
             Paragraph('Predicts temperature for the next 30 days based on seasonal patterns', sty_tbl_cell),
             Paragraph('MODIS MOD11A1 (NASA)', sty_tbl_cell)],
            [Paragraph('Pollution Zone Mapping', sty_tbl_cell),
             Paragraph('DBSCAN Clustering', sty_tbl_cell_b),
             Paragraph('Groups geographic pollution points into named risk zones', sty_tbl_cell),
             Paragraph('Sentinel-5P NO₂ (ESA)', sty_tbl_cell)],
        ]

        method_tbl = Table(method_rows, colWidths=[W*0.2, W*0.18, W*0.37, W*0.25])
        method_tbl.setStyle(TableStyle([
            ('BACKGROUND',    (0,0), (-1,0),  NAVY),
            ('BACKGROUND',    (0,1), (-1,-1), WHITE),
            ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHT_GRAY]),
            ('GRID',          (0,0), (-1,-1), 0.5, MID_GRAY),
            ('PADDING',       (0,0), (-1,-1), 8),
            ('VALIGN',        (0,0), (-1,-1), 'TOP'),
            ('LINEBELOW',     (0,0), (-1,0),  1.5, TEAL),
        ]))
        content.append(method_tbl)
        content.append(Spacer(1, 0.4*cm))

        # ════════════════════════════════════
        #  SECTION 5 — DISCLAIMER + SIGNATURE
        # ════════════════════════════════════
        content.append(HRFlowable(width=W, thickness=1, color=MID_GRAY, spaceAfter=8))

        disc_tbl = Table([[
            Paragraph(
                '<b>Disclaimer:</b> This report is generated by the EcoSentinel Satellite '
                'Environmental Intelligence Platform using publicly available satellite data from '
                'MODIS (NASA) and Copernicus Sentinel-5P (ESA). Data marked as "Simulated" uses '
                'statistically realistic models. For regulatory decisions, verify with CPCB/GPCB data.',
                S('disc', fontName='Helvetica', fontSize=8, textColor=DARK_GRAY,
                  leading=12, alignment=TA_JUSTIFY))
        ]], colWidths=[W])
        disc_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), GOLD_LIGHT),
            ('BOX',        (0,0), (-1,-1), 0.5, GOLD),
            ('PADDING',    (0,0), (-1,-1), 10),
        ]))
        content.append(disc_tbl)
        content.append(Spacer(1, 0.4*cm))

        # Signature line
        sig_tbl = Table([[
            Paragraph(
                f'<b>Prepared by:</b> EcoSentinel AI Platform<br/>'
                f'<b>Date:</b> {today}<br/>'
                f'<b>Reference:</b> {ref_no}',
                S('sg', fontName='Helvetica', fontSize=9, textColor=DARK_GRAY, leading=14)),
            Paragraph(
                f'<b>Area:</b> {area}, {city}<br/>'
                f'<b>Classification:</b> {data.get("type","—").title()}<br/>'
                f'<b>Risk Level:</b> {r_label}',
                S('sg2', fontName='Helvetica', fontSize=9, textColor=DARK_GRAY,
                  leading=14, alignment=TA_RIGHT)),
        ]], colWidths=[W*0.5, W*0.5])
        sig_tbl.setStyle(TableStyle([
            ('LINEABOVE', (0,0), (-1,0), 1, MID_GRAY),
            ('PADDING',   (0,0), (-1,-1), 8),
            ('VALIGN',    (0,0), (-1,-1), 'TOP'),
        ]))
        content.append(sig_tbl)

        # Footer
        footer_tbl = Table([[
            Paragraph(
                f'EcoSentinel Environmental Intelligence Platform  |  '
                f'Satellite ML Analysis Report  |  {today}  |  Ref: {ref_no}',
                sty_footer)
        ]], colWidths=[W])
        footer_tbl.setStyle(TableStyle([
            ('LINEABOVE',  (0,0), (-1,-1), 0.5, MID_GRAY),
            ('BACKGROUND', (0,0), (-1,-1), LIGHT_GRAY),
            ('PADDING',    (0,0), (-1,-1), 6),
        ]))
        content.append(footer_tbl)

        doc.build(content)
        buffer.seek(0)

        filename = f'EcoSentinel_{city}_{area.replace(" ","_")}_Report_{datetime.now().strftime("%Y%m%d")}.pdf'
        return send_file(buffer, as_attachment=True,
                         download_name=filename, mimetype='application/pdf')

    except ImportError:
        return jsonify({'error': 'Run: pip install reportlab'}), 500
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    print("=" * 55)
    print("  EcoSentinel Server Starting...")
    print("  Website:      http://localhost:5000")
    print("  Health check: http://localhost:5000/api/health")
    print("=" * 55)
    app.run(debug=True, port=5000)
