# utils/report_generator.py
from io import BytesIO
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepInFrame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle # Aggiunto ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
import pandas as pd
from datetime import datetime
# from reportlab.graphics. medziagomis import FigureCanvasAgg # Rimosso come discusso

def fig_to_image(fig, width=7*inch, height=3.5*inch):
    if fig is None:
        return None
    img_buffer = BytesIO()
    try:
        fig.savefig(img_buffer, format='PNG', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        return Image(img_buffer, width=width, height=height) 
    except Exception as e:
        print(f"Errore durante la conversione della figura Matplotlib in immagine: {e}")
        return None
    # Non chiudere la figura qui, verrà chiusa in main.py

def generate_pac_report_pdf(
    tickers_list,
    allocations_float_list_raw,
    pac_params,
    metrics_df, 
    asset_details_final_df,
    equity_line_fig,
    drawdown_fig,
    stacked_area_fig
):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=0.5*inch, leftMargin=0.5*inch,
                            topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    story = []

    # --- TITOLO ---
    title_style = styles['h1']
    title_style.alignment = 1 
    story.append(Paragraph("Report Simulazione Piano di Accumulo Capitale (PAC)", title_style))
    story.append(Spacer(1, 0.1*inch))
    time_style = styles['Normal']
    time_style.alignment = 1 
    story.append(Paragraph(f"Report generato il: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", time_style))
    story.append(Spacer(1, 0.2*inch))

    # --- PARAMETRI SIMULAZIONE (come prima) ---
    story.append(Paragraph("Parametri della Simulazione:", styles['h2']))
    param_text = f"<b>Asset e Allocazioni Target:</b><br/>"
    for i, ticker in enumerate(tickers_list): param_text += f"- {ticker}: {allocations_float_list_raw[i]:.1f}%<br/>"
    param_text += "<br/><b>Parametri PAC:</b><br/>"
    param_text += f"- Data Inizio Contributi: {pac_params['start_date']}<br/>"
    param_text += f"- Durata Contributi: {pac_params['duration_months']} mesi<br/>"
    param_text += f"- Importo Versamento Mensile: {pac_params['monthly_investment']}<br/>"
    param_text += f"- Reinvestimento Dividendi: {'Sì' if pac_params['reinvest_div'] else 'No'}<br/>"
    if pac_params['rebalance_active']: param_text += f"- Ribilanciamento Attivo: Sì, Frequenza: {pac_params['rebalance_freq']}<br/>"
    else: param_text += f"- Ribilanciamento Attivo: No<br/>"
    story.append(Paragraph(param_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))

    # --- TABELLA METRICHE DI PERFORMANCE (come prima) ---
    if metrics_df is not None and not metrics_df.empty:
        story.append(Paragraph("Metriche di Performance Riepilogative:", styles['h2']))
        # ... (codice per metrics_table_pdf come prima) ...
        header = ["Metrica"] + [col for col in metrics_df.columns if col != "Metrica"] 
        metrics_data_for_pdf = [header]
        for index, row in metrics_df.iterrows(): 
            metrics_data_for_pdf.append(row.tolist()) 
        num_value_cols = len(header) - 1
        col_widths = [2.5*inch] + [1.5*inch] * num_value_cols if num_value_cols > 0 else [sum([2.5, 1.5*num_value_cols])*inch] 
        metrics_table_pdf = Table(metrics_data_for_pdf, colWidths=col_widths)
        metrics_table_pdf.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,0), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 6), ('TOPPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.aliceblue), ('FONTSIZE', (0,1), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('LEFTPADDING', (0,0), (-1,-1), 2), ('RIGHTPADDING', (0,0), (-1,-1), 2) ]))
        story.append(metrics_table_pdf)
        story.append(Spacer(1, 0.2*inch))

    # --- GRAFICO EQUITY LINE ---
    img_equity = fig_to_image(equity_line_fig, width=7*inch, height=3.5*inch)
    if img_equity:
        story.append(Paragraph("Andamento Comparativo del Portafoglio:", styles['h2']))
        story.append(img_equity)
        story.append(Spacer(1, 0.1*inch))

    # --- GRAFICO DRAWDOWN ---
    img_drawdown = fig_to_image(drawdown_fig, width=7*inch, height=3.5*inch)
    if img_drawdown:
        story.append(Paragraph("Andamento del Drawdown nel Tempo:", styles['h2']))
        story.append(img_drawdown)
        story.append(Spacer(1, 0.1*inch))
    
    if img_equity or img_drawdown : story.append(PageBreak())

    # --- GRAFICO STACKED AREA ---
    img_stacked = fig_to_image(stacked_area_fig, width=7*inch, height=3.5*inch)
    if img_stacked:
        story.append(Paragraph("Allocazione Dinamica Portafoglio PAC:", styles['h2']))
        story.append(img_stacked)
        story.append(Spacer(1, 0.2*inch))

    # --- TABELLA DETTAGLI FINALI PER ASSET PAC ---
    if asset_details_final_df is not None and not asset_details_final_df.empty:
        story.append(Paragraph("Dettagli Finali per Asset nel PAC:", styles['h2']))
        # ... (codice per asset_table_pdf come prima) ...
        header_asset = asset_details_final_df.columns.tolist()
        asset_data_for_pdf = [header_asset]
        for index, row in asset_details_final_df.iterrows():
             asset_data_for_pdf.append(row.tolist())
        asset_table_pdf = Table(asset_data_for_pdf, colWidths=[1.5*inch, 1.5*inch, 2*inch, 2*inch])
        asset_table_pdf.setStyle(TableStyle([
             ('BACKGROUND', (0,0), (-1,0), colors.darkslateblue), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,0), 9),
            ('BOTTOMPADDING', (0,0), (-1,0), 6), ('TOPPADDING', (0,0), (-1,0), 6),
            ('BACKGROUND', (0,1), (-1,-1), colors.lightcyan), ('FONTSIZE', (0,1), (-1,-1), 8),
            ('GRID', (0,0), (-1,-1), 1, colors.black) ]))
        story.append(asset_table_pdf)
        story.append(Spacer(1, 0.2*inch))

    # --- AGGIUNTA LINK KRITERION QUANT NEL PDF ---
    story.append(Spacer(1, 0.5*inch)) # Un po' di spazio prima del link
    link_style = styles['Normal']
    link_style.alignment = 1 # Center
    # ReportLab interpreta i tag <a> per i link nei PDF
    kriterion_link_text = 'Visita il nostro sito: <a href="https://kriterionquant.com/" color="blue"><u>Kriterion Quant</u></a>'
    story.append(Paragraph(kriterion_link_text, link_style))
    story.append(Paragraph("<i>Progetto Didattico Kriterion Quant</i>", link_style)) # Aggiunto anche il nome del progetto

    try:
        doc.build(story)
        pdf_value = buffer.getvalue()
    except Exception as e_build:
        print(f"ERRORE ReportLab build: {e_build}") 
        buffer = BytesIO()
        doc_err = SimpleDocTemplate(buffer, pagesize=A4)
        story_err = [Paragraph("Errore durante la generazione del report PDF.", styles['h1']), Paragraph(str(e_build), styles['Normal'])]
        doc_err.build(story_err)
        pdf_value = buffer.getvalue()
        
    buffer.close()
    return pdf_value
