from flask import Flask, jsonify, render_template, send_file, request
from flask_cors import CORS
from functools import lru_cache
import io, os
from datetime import datetime

from ml_engine import analyze_area, analyze_city_overview, get_cities, get_areas

import os
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
CORS(app)


@lru_cache(maxsize=100)
def cached_analyze(city, area):
    return analyze_area(city, area)


# ══════════════════════════════════════════════
#  PAGE ROUTES
# ══════════════════════════════════════════════

# Bug_Slayers landing page — the beautiful map home screen
@app.route('/')
def home():
    return render_template('home.html')


# ML Dashboard — loaded after user selects city+area on home page
# URL looks like: /dashboard?city=Ahmedabad&area=Vatva
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
        return jsonify({'error': f'City {city} not found. Only Ahmedabad and Gandhinagar are supported.'}), 404
    return jsonify({'areas': area_list})


@app.route('/api/analyze/<city>/<area>')
def analyze(city, area):
    area = area.replace('%20', ' ').replace('+', ' ')
    # Fix common name mismatches that might still come through
    name_map = {
        'Narangpura': 'Naranpura',
        'Ambavadi':   'Ambawadi',
    }
    area = name_map.get(area, area)
    try:
        result = cached_analyze(city, area)
        if result is None:
            return jsonify({'error': f'{area} not found in {city}'}), 404
        return jsonify(result)
    except KeyError:
        return jsonify({'error': f'{area} not found in {city}. Check area name spelling.'}), 404
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
        'status': 'running',
        'cities': get_cities(),
        'message': 'EcoSentinel is live',
        'routes': {
            '/':          'Landing page (Bug_Slayers map)',
            '/dashboard': 'ML Dashboard (pass ?city=X&area=Y)',
        }
    })


# ══════════════════════════════════════════════
#  AI ADVICE
# ══════════════════════════════════════════════

def get_ai_advice(data):
    try:
        import anthropic
        client  = anthropic.Anthropic()
        prompt  = f"""You are an environmental advisor for Indian municipal corporations.
Analyze this satellite ML data for {data['area']}, {data['city']} and give practical advice.

AREA DATA:
- Area type: {data['type']}
- Average temperature: {data['avg_temp']}°C
- Average NO2 pollution: {data['avg_no2']} μg/m³
- Unusual temperature days found: {data['anomaly_count']} days in 2023
- Worst temperature recorded: {data.get('worst_anomaly_temp', 'N/A')}°C
- 30-day forecast: {data['forecast_avg']}°C (trend: {data['trend']})
- Pollution zones found: {data['hotspot_clusters']}
- Habitability score: {data['habitability']['score']}/100 ({data['habitability']['label']})
- Overall risk: {data['risk']['label']} (score: {data['risk']['score']}/100)

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
        # Rule-based fallback
        risk  = data['risk']['score']
        area  = data['area']
        temp  = data['avg_temp']
        no2   = data['avg_no2']
        hab   = data['habitability']['score']

        if risk > 70:
            return f"""1. CURRENT SITUATION
{area} is in HIGH RISK state. Temperature of {temp}°C with NO2 at {no2} ug/m3 requires immediate government action. {data['anomaly_count']} dangerous temperature days were detected in 2023.

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
AMC Environment Department + Gujarat Pollution Control Board (GPCB) + Local factory owners association.

5. EXPECTED IMPROVEMENT
NO2 pollution can reduce by 20-30% in 90 days. Habitability score can improve from {hab} to above 60 within 3 months."""

        elif risk > 40:
            return f"""1. CURRENT SITUATION
{area} is at MODERATE RISK. NO2 at {no2} ug/m3 exceeds the safe limit of 40 ug/m3. Consistent monitoring required.

2. IMMEDIATE ACTIONS NEEDED
- Increase green cover by planting trees along all main roads
- Set up 2 permanent air quality monitoring stations
- Enforce vehicle pollution checks at key intersections
- Promote rooftop garden initiatives in housing societies

3. 30-DAY ACTION PLAN
Week 1: Install monitors, record baseline measurements.
Week 2-3: Tree plantation drive with Resident Welfare Associations.
Week 4: Review data, publish public environmental status report.

4. WHO SHOULD ACT
Ward-level AMC officers + Resident Welfare Associations (RWAs) + Local industries.

5. EXPECTED IMPROVEMENT
Habitability score can improve from {hab} to above 70 within 60 days of action."""

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
    area = {'Narangpura':'Naranpura','Ambavadi':'Ambawadi'}.get(area, area)
    try:
        data   = cached_analyze(city, area)
        advice = get_ai_advice(data)
        return jsonify({'advice': advice})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ══════════════════════════════════════════════
#  PDF REPORT GENERATION
# ══════════════════════════════════════════════

@app.route('/api/report/<city>/<area>')
def generate_report(city, area):
    area = area.replace('%20', ' ').replace('+', ' ')
    area = {'Narangpura':'Naranpura','Ambavadi':'Ambawadi'}.get(area, area)
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
        import re

        data   = cached_analyze(city, area)
        risk   = data.get('risk', {})
        hab    = data.get('habitability', {})
        today  = datetime.now().strftime("%d %B %Y")
        ref_no = f"ECO/{city[:3].upper()}/{datetime.now().strftime('%Y%m%d')}/001"

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
        RED_ALERT   = colors.HexColor('#C53030')
        ORANGE_WARN = colors.HexColor('#C05621')
        GREEN_GOOD  = colors.HexColor('#276749')

        def rc(s): return RED_ALERT if s>70 else ORANGE_WARN if s>40 else GREEN_GOOD
        def no2s(v): return ("Hazardous",RED_ALERT) if v>80 else ("Unhealthy",ORANGE_WARN) if v>50 else ("Moderate",GOLD) if v>30 else ("Good",GREEN_GOOD)
        def ts(v):   return ("Critical",RED_ALERT) if v>35 else ("Elevated",ORANGE_WARN) if v>28 else ("Normal",GREEN_GOOD)
        def S(n,**k): return ParagraphStyle(n,**k)

        sh = S('sh',fontName='Helvetica-Bold',fontSize=12,textColor=TEAL,leading=16,spaceBefore=14,spaceAfter=6,alignment=TA_LEFT)
        sb = S('b', fontName='Helvetica',fontSize=10,textColor=DARK_GRAY,leading=16,spaceAfter=4,alignment=TA_JUSTIFY)
        sbb= S('bb',fontName='Helvetica-Bold',fontSize=10,textColor=BLACK,leading=16,spaceAfter=4)
        sf = S('ft',fontName='Helvetica',fontSize=8,textColor=colors.HexColor('#94A3B8'),alignment=TA_CENTER)
        th = S('th',fontName='Helvetica-Bold',fontSize=9,textColor=WHITE,leading=12,alignment=TA_LEFT)
        tc = S('tc',fontName='Helvetica',fontSize=9,textColor=DARK_GRAY,leading=13,alignment=TA_LEFT)
        tcb= S('tcb',fontName='Helvetica-Bold',fontSize=9,textColor=BLACK,leading=13)
        ah = S('ah',fontName='Helvetica-Bold',fontSize=10,textColor=NAVY,leading=14,spaceBefore=8,spaceAfter=2)
        ab = S('ab',fontName='Helvetica',fontSize=10,textColor=DARK_GRAY,leading=15,spaceAfter=3,leftIndent=12,alignment=TA_JUSTIFY)

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=1.5*cm, bottomMargin=2*cm,
                                title=f"EcoSentinel Report — {area}, {city}",
                                author="EcoSentinel")
        W   = 17*cm
        cnt = []

        # Banner
        banner = Table([[Paragraph(
            '<font size="8" color="#A0B4C8">ECOSENTINEL · SATELLITE ENVIRONMENTAL INTELLIGENCE PLATFORM · CONFIDENTIAL</font>',
            S('b2',fontName='Helvetica',fontSize=8,textColor=colors.HexColor('#A0B4C8'),alignment=TA_CENTER)
        )]], colWidths=[W])
        banner.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),DARK_NAVY),('PADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,-1),2,GOLD)]))
        cnt.append(banner)
        cnt.append(Spacer(1,0.8*cm))

        # Logo row
        lr = Table([[
            Paragraph('<b>ECOSENTINEL · GREENGRID</b>',S('lg',fontName='Helvetica-Bold',fontSize=18,textColor=TEAL)),
            Paragraph('<font color="#94A3B8" size="8">Powered by MODIS · Sentinel-5P · Isolation Forest · ARIMA · DBSCAN</font>',
                      S('pl',fontName='Helvetica',fontSize=8,textColor=colors.HexColor('#94A3B8'),alignment=TA_RIGHT)),
        ]],colWidths=[W*0.5,W*0.5])
        lr.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('PADDING',(0,0),(-1,-1),0)]))
        cnt.append(lr)
        cnt.append(HRFlowable(width=W,thickness=1,color=MID_GRAY,spaceAfter=10))

        # Title block
        tb = Table([[Table([
            [Paragraph('ENVIRONMENTAL INTELLIGENCE REPORT',S('rt',fontName='Helvetica-Bold',fontSize=9,textColor=GOLD,leading=12))],
            [Paragraph(area.upper(),S('an',fontName='Helvetica-Bold',fontSize=28,textColor=DARK_NAVY,leading=32))],
            [Paragraph(f'{city}, Gujarat, India',S('cn',fontName='Helvetica-Bold',fontSize=14,textColor=TEAL,leading=18))],
            [Spacer(1,0.3*cm)],
            [Paragraph('Satellite-based environmental analysis using Machine Learning. Data sourced from MODIS NASA and Copernicus Sentinel-5P.',
                       S('in',fontName='Helvetica',fontSize=10,textColor=DARK_GRAY,leading=15,alignment=TA_JUSTIFY))],
        ],colWidths=[W*0.65])]],colWidths=[W])
        tb.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),LIGHT_GRAY),('PADDING',(0,0),(-1,-1),20)]))
        cnt.append(tb)
        cnt.append(Spacer(1,0.4*cm))

        # Meta row
        mt = Table([[
            Paragraph(f'<b>Reference No.</b><br/>{ref_no}',S('mi',fontName='Helvetica',fontSize=9,textColor=DARK_GRAY,leading=13)),
            Paragraph(f'<b>Date of Report</b><br/>{today}',S('mi2',fontName='Helvetica',fontSize=9,textColor=DARK_GRAY,leading=13)),
            Paragraph(f'<b>Area Classification</b><br/>{data.get("type","—").title()}',S('mi3',fontName='Helvetica',fontSize=9,textColor=DARK_GRAY,leading=13)),
            Paragraph(f'<b>Data Source</b><br/>{data.get("data_source","Simulated")}',S('mi4',fontName='Helvetica',fontSize=9,textColor=DARK_GRAY,leading=13)),
        ]],colWidths=[W/4]*4)
        mt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),WHITE),('BOX',(0,0),(-1,-1),1,MID_GRAY),('INNERGRID',(0,0),(-1,-1),0.5,MID_GRAY),('PADDING',(0,0),(-1,-1),10),('VALIGN',(0,0),(-1,-1),'TOP')]))
        cnt.append(mt)
        cnt.append(Spacer(1,0.5*cm))

        # 4 metric cards
        rs = risk.get('score',0); rl = risk.get('label','—')
        hs = hab.get('score',0);  hl = hab.get('label','—')
        tstat,tcol = ts(data.get('avg_temp',0))
        nstat,ncol = no2s(data.get('avg_no2',0))
        def mc(title,val,unit,lbl,col):
            return Table([
                [Paragraph(title,S('mct',fontName='Helvetica-Bold',fontSize=8,textColor=colors.HexColor('#64748B'),leading=10))],
                [Paragraph(f'<font size="22" color="{col.hexval()}"><b>{val}</b></font><font size="10" color="#64748B"> {unit}</font>',S('mcv',fontName='Helvetica',fontSize=22,leading=26))],
                [Paragraph(lbl,S('mcl',fontName='Helvetica-Bold',fontSize=8,textColor=col,leading=10))],
            ],colWidths=[(W/4)-0.3*cm])
        cr = [mc("RISK SCORE",str(rs),"/ 100",rl,rc(rs)),
              mc("HABITABILITY",str(hs),"/ 100",hl,GREEN_GOOD if hs>60 else ORANGE_WARN if hs>40 else RED_ALERT),
              mc("AVG TEMPERATURE",str(data.get('avg_temp','—')),"°C",tstat,tcol),
              mc("AVG NO₂ POLLUTION",str(data.get('avg_no2','—')),"μg/m³",nstat,ncol)]
        ct = Table([cr],colWidths=[W/4]*4)
        ct.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),WHITE),('BOX',(0,0),(-1,-1),1,MID_GRAY),('INNERGRID',(0,0),(-1,-1),0.5,MID_GRAY),('PADDING',(0,0),(-1,-1),12),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,-1),3,TEAL)]))
        cnt.append(ct)
        cnt.append(Spacer(1,0.6*cm))

        # Section 1 — Executive Summary
        cnt.append(HRFlowable(width=W,thickness=2,color=TEAL,spaceAfter=6))
        cnt.append(Paragraph('1. EXECUTIVE SUMMARY',sh))
        cnt.append(Paragraph(
            f"This report presents a comprehensive environmental analysis of <b>{area}</b>, {city}, Gujarat, "
            f"using satellite-derived data and Machine Learning. The area is classified as <b>{data.get('type','mixed').title()}</b> "
            f"with risk score <b>{rs}/100</b> ({rl}) and habitability <b>{hs}/100</b> ({hl}).",sb))
        cnt.append(Paragraph(
            f"Average land surface temperature: <b>{data.get('avg_temp','—')}°C</b>. "
            f"NO₂: <b>{data.get('avg_no2','—')} μg/m³</b> ({nstat}). "
            f"Isolation Forest detected <b>{data.get('anomaly_count',0)} unusual days</b>. "
            f"ARIMA projects <b>{data.get('forecast_avg','—')}°C</b> avg next 30 days, trend: <b>{data.get('trend','Stable')}</b>.",sb))
        cnt.append(Spacer(1,0.3*cm))

        # Section 2 — ML Results
        cnt.append(HRFlowable(width=W,thickness=2,color=TEAL,spaceAfter=6))
        cnt.append(Paragraph('2. MACHINE LEARNING ANALYSIS RESULTS',sh))
        cnt.append(Paragraph('2.1  Environmental Parameters',sbb))
        pr = [
            [Paragraph('Parameter',th),Paragraph('Measured Value',th),Paragraph('Status',th),Paragraph('Threshold',th)],
            [Paragraph('Land Surface Temperature (Avg)',tc),Paragraph(f"{data.get('avg_temp','—')}°C",tcb),Paragraph(tstat,S('s1',fontName='Helvetica-Bold',fontSize=9,textColor=tcol)),Paragraph('Safe < 28°C',tc)],
            [Paragraph('Land Surface Temperature (Max)',tc),Paragraph(f"{data.get('max_temp','—')}°C",tcb),Paragraph('Recorded Peak',tc),Paragraph('Alert > 42°C',tc)],
            [Paragraph('NO₂ Concentration (Avg)',tc),Paragraph(f"{data.get('avg_no2','—')} μg/m³",tcb),Paragraph(nstat,S('s2',fontName='Helvetica-Bold',fontSize=9,textColor=ncol)),Paragraph('Safe < 40 μg/m³',tc)],
            [Paragraph('ARIMA 30-Day Forecast',tc),Paragraph(f"{data.get('forecast_avg','—')}°C",tcb),Paragraph(data.get('trend','Stable'),tc),Paragraph('Next 30 days',tc)],
            [Paragraph('Anomaly Days (Isolation Forest)',tc),Paragraph(str(data.get('anomaly_count',0)),tcb),Paragraph('Detected',tc),Paragraph('Ideal: 0 days',tc)],
            [Paragraph('Pollution Zones (DBSCAN)',tc),Paragraph(str(data.get('hotspot_clusters',0)),tcb),Paragraph('Identified',tc),Paragraph('Ideal: < 2 zones',tc)],
            [Paragraph('Habitability Index',tc),Paragraph(f"{hs}/100",tcb),Paragraph(hl,S('s3',fontName='Helvetica-Bold',fontSize=9,textColor=GREEN_GOOD if hs>60 else ORANGE_WARN if hs>40 else RED_ALERT)),Paragraph('Good > 60',tc)],
            [Paragraph('Overall Risk Score',tc),Paragraph(f"{rs}/100",tcb),Paragraph(rl,S('s4',fontName='Helvetica-Bold',fontSize=9,textColor=rc(rs))),Paragraph('Low < 40',tc)],
        ]
        pt = Table(pr,colWidths=[W*0.35,W*0.2,W*0.22,W*0.23])
        pt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('TEXTCOLOR',(0,0),(-1,0),WHITE),('BACKGROUND',(0,1),(-1,-1),WHITE),('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,LIGHT_GRAY]),('GRID',(0,0),(-1,-1),0.5,MID_GRAY),('PADDING',(0,0),(-1,-1),8),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LINEBELOW',(0,0),(-1,0),1.5,TEAL)]))
        cnt.append(pt)
        cnt.append(Spacer(1,0.4*cm))

        # Anomaly table
        anomalies = data.get('anomalies',[])
        if anomalies:
            cnt.append(Paragraph(f'2.2  Unusual Temperature Days ({len(anomalies)} anomalies identified)',sbb))
            ar = [[Paragraph('Date',th),Paragraph('Temperature',th),Paragraph('Deviation from Normal',th),Paragraph('Severity',th)]]
            avgt = data.get('avg_temp',30)
            for a in anomalies[:12]:
                dev = round(a.get('temp',avgt)-avgt,1)
                ds  = f"+{dev}°C above normal" if dev>0 else f"{abs(dev)}°C below normal"
                sev = a.get('severity','Low')
                sc  = RED_ALERT if sev=='High' else ORANGE_WARN if sev=='Medium' else GREEN_GOOD
                sl  = 'Very Unusual' if sev=='High' else 'Unusual' if sev=='Medium' else 'Slightly Unusual'
                ar.append([Paragraph(a.get('date','—'),tc),Paragraph(f"{a.get('temp','—')}°C",tcb),Paragraph(ds,S('ds',fontName='Helvetica',fontSize=9,textColor=RED_ALERT if dev>0 else TEAL)),Paragraph(sl,S('sl',fontName='Helvetica-Bold',fontSize=9,textColor=sc))])
            at = Table(ar,colWidths=[W*0.22,W*0.18,W*0.35,W*0.25])
            at.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('BACKGROUND',(0,1),(-1,-1),WHITE),('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,LIGHT_GRAY]),('GRID',(0,0),(-1,-1),0.5,MID_GRAY),('PADDING',(0,0),(-1,-1),7),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LINEBELOW',(0,0),(-1,0),1.5,TEAL)]))
            cnt.append(at)
            cnt.append(Spacer(1,0.3*cm))

        # Section 3 — AI Action Plan
        cnt.append(HRFlowable(width=W,thickness=2,color=TEAL,spaceAfter=6))
        cnt.append(Paragraph('3. AI-GENERATED ENVIRONMENTAL ACTION PLAN',sh))
        cnt.append(Paragraph(f'Generated by Claude AI (Anthropic) based on satellite ML data for {area}, {city}.',sb))
        cnt.append(Spacer(1,0.2*cm))
        try: advice_text = get_ai_advice(data)
        except: advice_text = "AI advice unavailable."
        adv_paras = []
        for line in advice_text.strip().split('\n'):
            line = line.strip()
            if not line: continue
            if re.match(r'^\d\.', line): adv_paras.append(Paragraph(line,ah))
            elif line.startswith(('-','•')): adv_paras.append(Paragraph('• '+line.lstrip('-•').strip(),ab))
            elif line.startswith('Week'): adv_paras.append(Paragraph(f'<b>{line}</b>',ab))
            else: adv_paras.append(Paragraph(line,ab))
        if adv_paras:
            adv_t = Table([[p] for p in adv_paras],colWidths=[W-0.8*cm])
            adv_t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),TEAL_LIGHT),('PADDING',(0,0),(-1,-1),4),('LEFTPADDING',(0,0),(-1,-1),16),('BOX',(0,0),(-1,-1),1.5,TEAL),('LINEAFTER',(0,0),(0,-1),3,GOLD)]))
            cnt.append(adv_t)
        cnt.append(Spacer(1,0.5*cm))

        # Section 4 — Methodology
        cnt.append(HRFlowable(width=W,thickness=2,color=TEAL,spaceAfter=6))
        cnt.append(Paragraph('4. METHODOLOGY AND DATA SOURCES',sh))
        mr = [
            [Paragraph('ML Model',th),Paragraph('Algorithm',th),Paragraph('Purpose',th),Paragraph('Data Source',th)],
            [Paragraph('Anomaly Detection',tc),Paragraph('Isolation Forest',tcb),Paragraph('Finds unusual temperature days — heat waves and cold snaps',tc),Paragraph('MODIS MOD11A1 (NASA)',tc)],
            [Paragraph('Temperature Forecast',tc),Paragraph('ARIMA (3,1,0)',tcb),Paragraph('Predicts temperature for next 30 days based on seasonal patterns',tc),Paragraph('MODIS MOD11A1 (NASA)',tc)],
            [Paragraph('Pollution Zone Mapping',tc),Paragraph('DBSCAN Clustering',tcb),Paragraph('Groups geographic pollution points into named risk zones',tc),Paragraph('Sentinel-5P NO₂ (ESA)',tc)],
        ]
        mrt = Table(mr,colWidths=[W*0.2,W*0.18,W*0.37,W*0.25])
        mrt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('BACKGROUND',(0,1),(-1,-1),WHITE),('ROWBACKGROUNDS',(0,1),(-1,-1),[WHITE,LIGHT_GRAY]),('GRID',(0,0),(-1,-1),0.5,MID_GRAY),('PADDING',(0,0),(-1,-1),8),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),1.5,TEAL)]))
        cnt.append(mrt)
        cnt.append(Spacer(1,0.4*cm))

        # Disclaimer
        cnt.append(HRFlowable(width=W,thickness=1,color=MID_GRAY,spaceAfter=8))
        dt = Table([[Paragraph('<b>Disclaimer:</b> Generated by EcoSentinel using publicly available satellite data from MODIS (NASA) and Copernicus Sentinel-5P (ESA). For regulatory decisions, verify with CPCB/GPCB official data.',S('disc',fontName='Helvetica',fontSize=8,textColor=DARK_GRAY,leading=12,alignment=TA_JUSTIFY))]],colWidths=[W])
        dt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),GOLD_LIGHT),('BOX',(0,0),(-1,-1),0.5,GOLD),('PADDING',(0,0),(-1,-1),10)]))
        cnt.append(dt)
        cnt.append(Spacer(1,0.4*cm))

        # Signature
        sig = Table([[
            Paragraph(f'<b>Prepared by:</b> EcoSentinel AI Platform<br/><b>Date:</b> {today}<br/><b>Reference:</b> {ref_no}',S('sg',fontName='Helvetica',fontSize=9,textColor=DARK_GRAY,leading=14)),
            Paragraph(f'<b>Area:</b> {area}, {city}<br/><b>Classification:</b> {data.get("type","—").title()}<br/><b>Risk Level:</b> {rl}',S('sg2',fontName='Helvetica',fontSize=9,textColor=DARK_GRAY,leading=14,alignment=TA_RIGHT)),
        ]],colWidths=[W*0.5,W*0.5])
        sig.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),1,MID_GRAY),('PADDING',(0,0),(-1,-1),8),('VALIGN',(0,0),(-1,-1),'TOP')]))
        cnt.append(sig)

        # Footer
        ft = Table([[Paragraph(f'EcoSentinel · GreenGrid Environmental Intelligence Platform  |  {today}  |  Ref: {ref_no}',sf)]],colWidths=[W])
        ft.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,-1),0.5,MID_GRAY),('BACKGROUND',(0,0),(-1,-1),LIGHT_GRAY),('PADDING',(0,0),(-1,-1),6)]))
        cnt.append(ft)

        doc.build(cnt)
        buf.seek(0)
        fname = f'EcoSentinel_{city}_{area.replace(" ","_")}_Report_{datetime.now().strftime("%Y%m%d")}.pdf'
        return send_file(buf,as_attachment=True,download_name=fname,mimetype='application/pdf')

    except ImportError:
        return jsonify({'error':'Run: pip install reportlab'}),500
    except Exception as e:
        import traceback
        return jsonify({'error':str(e),'trace':traceback.format_exc()}),500


if __name__ == '__main__':
    print("=" * 60)
    print("  EcoSentinel + GreenGrid Server Starting...")
    print("  Home (Bug_Slayers map): http://localhost:5000")
    print("  ML Dashboard:           http://localhost:5000/dashboard")
    print("  Health check:           http://localhost:5000/api/health")
    print("=" * 60)
    app.run(debug=True, port=5000)
