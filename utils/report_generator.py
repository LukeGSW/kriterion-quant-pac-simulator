# utils/report_generator.py
from io import BytesIO
from reportlab.lib.pagesizes import A4, landscape # Aggiunto landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepInFrame
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
import pandas as pd
from datetime import datetime
# Per integrare figure Matplotlib
from reportlab.graphics. medziagomis import FigureCanvasAgg
# from reportlab.graphics import renderPDF # Non necessario se usiamo Image

def fig_to_image(fig):
    """Converte una figura Matplotlib in un oggetto Image di ReportLab."""
    if fig is None:
        return None
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='PNG', dpi=150) # dpi più basso per file PDF più leggeri
    img_buffer.seek(0)
    # plt.close(fig) # Chiudi la figura per liberare memoria se non serve più
    return Image(img_buffer, width=7*inch, height=3.5*inch) # Ajusta dimensioni come necessario

def generate_pac_report_pdf(
    tickers_list,
    allocations_float_list_raw,
    pac_params,
    metrics_df, # DataFrame della tabella delle metriche (con indice "Metrica")
    # pac_total_df, # Non più necessario direttamente se abbiamo i grafici
    # lump_sum_df,
    asset_details_final_df, # DataFrame con Quote Finali e WAP per asset PAC (con indice "Ticker")
    equity_line_fig,
    drawdown_fig,
    stacked_area_fig
):
    buffer = BytesIO()
    # Usiamo due margini diversi per permettere tabelle più larghe e grafici
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=0.5*inch, leftMargin=0.5*inch,
                            topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    styles = getSampleStyleSheet()
    story = []

    # --- TITOLO ---
    title_style = styles['h1']
    title_style.alignment = 1 # Center
    story.append(Paragraph("Report Simulazione Piano di Accumulo Capitale (PAC)", title_style))
    story.append(Spacer(1, 0.1*inch))
    time_style = styles['Normal']
    time_style.alignment = 1 # Center
    story.append(Paragraph(f"Report generato il: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", time_style))
    story.append(Spacer(1, 0.2*inch))

    # --- PARAMETRI SIMULAZIONE (come prima) ---
    story.append(Paragraph("Parametri della Simulazione:", styles['h2']))
    # ... (codice per param_text come prima) ...
    param_text = f"<b>Asset e Allocazioni Target:</b><br/>"
    for i, ticker in enumerate(tickers_list): param_text += f"- {ticker}: {allocations_float_list_raw[i]:.1f}%<br/>"
    param_text += "<br/><b>Parametri PAC:</b><br/>"
    param_text += f"- Data Inizio Contributi: {pac_params['start_date']}<br/>" # ... (resto dei parametri)
    param_text += f"- Durata Contributi: {pac_params['duration_months']} mesi<br/>"
    param_text += f"- Importo Versamento Mensile: {pac_params['monthly_investment']}<br/>"
    param_text += f"- Reinvestimento Dividendi: {'Sì' if pac_params['reinvest_div'] else 'No'}<br/>"
    if pac_params['rebalance_active']: param_text += f"- Ribilanciamento Attivo: Sì, Frequenza: {pac_params['rebalance_freq']}<br/>"
    else: param_text += f"- Ribilanciamento Attivo: No<br/>"
    story.append(Paragraph(param_text, styles['Normal']))
    story.append(Spacer(1, 0.2*inch))


    # --- TABELLA METRICHE DI PERFORMANCE ---
    if metrics_df is not None and not metrics_df.empty:
        story.append(Paragraph("Metriche di Performance Riepilogative:", styles['h2']))
        # ... (codice per metrics_table come prima, ma aggiusta larghezze colonne se necessario) ...
        # Header: Metrica, PAC, Lump Sum (se esiste)
        header = ["Metrica"] + [col for col in metrics_df.columns if col != "Metrica"] # Prende le colonne PAC, LS
        metrics_data_for_pdf = [header]
        for index, row in metrics_df.iterrows(): # metrics_df ora ha "Metrica" come colonna normale
            metrics_data_for_pdf.append([row["Metrica"]] + [row[col_name] for col_name in header if col_name != "Metrica"])

        # Calcola larghezze colonne: prima colonna più larga, le altre uguali
        num_value_cols = len(header) - 1
        col_widths = [2.5*inch] + [1.5*inch] * num_value_cols if num_value_cols > 0 else [4*inch]
        
        metrics_table_pdf = Table(metrics_data_for_pdf, colWidths=col_widths)
        metrics_table_pdf.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.darkblue), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.aliceblue), ('FONTSIZE', (0,1), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
            ('LEFTPADDING', (0,0), (-1,-1), 3), ('RIGHTPADDING', (0,0), (-1,-1), 3)
        ]))
        story.append(metrics_table_pdf)
        story.append(Spacer(1, 0.2*inch))


    # --- GRAFICO EQUITY LINE ---
    img_equity = fig_to_image(equity_line_fig)
    if img_equity:
        story.append(Paragraph("Andamento Comparativo del Portafoglio:", styles['h2']))
        story.append(img_equity)
        story.append(Spacer(1, 0.1*inch))

    # --- GRAFICO DRAWDOWN ---
    img_drawdown = fig_to_image(drawdown_fig)
    if img_drawdown:
        story.append(Paragraph("Andamento del Drawdown nel Tempo:", styles['h2']))
        story.append(img_drawdown)
        story.append(Spacer(1, 0.1*inch))
        story.append(PageBreak()) # Interruzione di pagina dopo i primi due grafici


    # --- GRAFICO STACKED AREA ---
    img_stacked = fig_to_image(stacked_area_fig)
    if img_stacked:
        story.append(Paragraph("Allocazione Dinamica Portafoglio PAC:", styles['h2']))
        story.append(img_stacked)
        story.append(Spacer(1, 0.2*inch))


    # --- TABELLA DETTAGLI FINALI PER ASSET PAC ---
    if asset_details_final_df is not None and not asset_details_final_df.empty:
        story.append(Paragraph("Dettagli Finali per Asset nel PAC:", styles['h2']))
        # ... (codice per asset_table come prima, ma aggiusta larghezze colonne) ...
        header_asset = ["Ticker"] + [col for col in asset_details_final_df.columns if col != "Ticker"]
        asset_data_for_pdf = [header_asset]
        for index, row in asset_details_final_df.iterrows():
            asset_data_for_pdf.append([row["Ticker"]] + [row[col_name] for col_name in header_asset if col_name != "Ticker"])
        
        asset_table_pdf = Table(asset_data_for_pdf, colWidths=[1.5*inch, 1.5*inch, 2*inch, 2*inch]) # Adatta larghezze
        asset_table_pdf.setStyle(TableStyle([
             ('BACKGROUND', (0,0), (-1,0), colors.darkslateblue), ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke), # Colore diverso
            ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,0), 8),
            ('BACKGROUND', (0,1), (-1,-1), colors.lightcyan), ('FONTSIZE', (0,1), (-1,-1), 9),
            ('GRID', (0,0), (-1,-1), 1, colors.black)
        ]))
        story.append(asset_table_pdf)
        story.append(Spacer(1, 0.2*inch))

    try:
        doc.build(story)
        pdf_value = buffer.getvalue()
    except Exception as e_build:
        print(f"ERRORE ReportLab build: {e_build}") # Aggiungi questo per debug
        # Crea un PDF di errore semplice se il build fallisce
        buffer = BytesIO()
        doc_err = SimpleDocTemplate(buffer, pagesize=A4)
        story_err = [Paragraph("Errore durante la generazione del report PDF.", styles['h1']),
                     Paragraph(str(e_build), styles['Normal'])]
        doc_err.build(story_err)
        pdf_value = buffer.getvalue()
        
    buffer.close()
    # if equity_line_fig: plt.close(equity_line_fig) # Chiudi figure per liberare memoria
    # if drawdown_fig: plt.close(drawdown_fig)
    # if stacked_area_fig: plt.close(stacked_area_fig)
    return pdf_value
