# utils/report_generator.py
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
import pandas as pd
from datetime import datetime

# Importa le figure Matplotlib se le generi altrove e le passi
# Oppure genera le figure Matplotlib direttamente qui dentro
# from reportlab.graphics.shapes import Drawing
# from reportlab.graphics.charts.lineplots import LinePlot
# from reportlab.graphics.charts.axes import XValueAxis, YValueAxis
# from reportlab.graphics.charts.legends import Legend
# from reportlab.lib.colors import PCMYKColor, black

def generate_pac_report_pdf(
    tickers_list,
    allocations_float_list_raw, # Es. [60.0, 20.0, 20.0]
    pac_params, # Dizionario con: start_date, duration_months, monthly_investment, reinvest_div, rebalance_active, rebalance_freq
    metrics_df, # DataFrame della tabella delle metriche (indice 'Metrica', colonne 'PAC', 'Lump Sum')
    pac_total_df, # DataFrame dell'evoluzione del PAC
    lump_sum_df,  # DataFrame dell'evoluzione del Lump Sum
    asset_details_final_df # DataFrame con Quote Finali e WAP per asset PAC
    # Aggiungeremo qui i parametri per le figure Matplotlib in seguito
    # equity_chart_image_buffer, # Esempio
    # drawdown_chart_image_buffer,
    # stacked_area_chart_image_buffer
):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    story = []

    # --- TITOLO ---
    title = "Report Simulazione Piano di Accumulo Capitale (PAC)"
    story.append(Paragraph(title, styles['h1']))
    story.append(Spacer(1, 0.2*inch))

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph(f"Report generato il: {current_time}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # --- PARAMETRI SIMULAZIONE ---
    story.append(Paragraph("Parametri della Simulazione:", styles['h2']))
    story.append(Spacer(1, 0.1*inch))
    
    param_text = f"<b>Asset e Allocazioni Target:</b><br/>"
    for i, ticker in enumerate(tickers_list):
        param_text += f"- {ticker}: {allocations_float_list_raw[i]:.1f}%<br/>"
    param_text += "<br/>"
    
    param_text += f"<b>Parametri PAC:</b><br/>"
    param_text += f"- Data Inizio Contributi: {pac_params['start_date']}<br/>"
    param_text += f"- Durata Contributi: {pac_params['duration_months']} mesi<br/>"
    param_text += f"- Importo Versamento Mensile: {pac_params['monthly_investment']}<br/>"
    param_text += f"- Reinvestimento Dividendi: {'Sì' if pac_params['reinvest_div'] else 'No'}<br/>"
    if pac_params['rebalance_active']:
        param_text += f"- Ribilanciamento Attivo: Sì, Frequenza: {pac_params['rebalance_freq']}<br/>"
    else:
        param_text += f"- Ribilanciamento Attivo: No<br/>"
    story.append(Paragraph(param_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # --- TABELLA METRICHE DI PERFORMANCE ---
    if metrics_df is not None and not metrics_df.empty:
        story.append(Paragraph("Metriche di Performance Riepilogative:", styles['h2']))
        story.append(Spacer(1, 0.1*inch))
        
        # Prepara i dati per la tabella ReportLab
        metrics_data = [metrics_df.columns.tolist()] # Header
        for index, row in metrics_df.iterrows():
            metrics_data.append([index] + row.tolist()) # Aggiungi nome metrica come prima colonna

        # Crea la tabella
        metrics_table = Table(metrics_data)
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.beige),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 0.2*inch))

    # --- TABELLA DETTAGLI FINALI PER ASSET PAC ---
    if asset_details_final_df is not None and not asset_details_final_df.empty:
        story.append(Paragraph("Dettagli Finali per Asset nel PAC:", styles['h2']))
        story.append(Spacer(1, 0.1*inch))

        asset_data_table = [asset_details_final_df.columns.tolist()] # Header
        for index, row in asset_details_final_df.iterrows():
             asset_data_table.append([index] + row.tolist()) # Aggiungi ticker come prima colonna

        asset_table = Table(asset_data_table)
        asset_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.lightyellow), # Colore diverso per questa tabella
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(asset_table)
        story.append(Spacer(1, 0.2*inch))

    # --- QUI AGGIUNGEREMO I GRAFICI ---
    # Per ora, un placeholder
    story.append(Paragraph("Grafici (placeholder):", styles['h2']))
    story.append(Paragraph("Il grafico dell'equity line, drawdown, ecc. verranno inseriti qui.", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))


    doc.build(story)
    pdf_value = buffer.getvalue()
    buffer.close()
    return pdf_value
