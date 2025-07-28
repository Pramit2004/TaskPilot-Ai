# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# from reportlab.lib.units import inch
# from typing import List, Dict, Any
# import os
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
# from reportlab.lib.styles import getSampleStyleSheet


# class ReportAgent:
#     """
#     Agent to generate end-to-end reports (PDF) with EDA, model metrics, and insights using ReportLab.
#     """

#     def generate_report(self, insights: str, charts: List[str], logs: Dict[str, Any], output_path: str = "reports/final_report.pdf") -> str:
#         os.makedirs(os.path.dirname(output_path), exist_ok=True)

#         # Create a PDF doc
#         doc = SimpleDocTemplate(output_path, pagesize=letter)
#         styles = getSampleStyleSheet()
#         flowables = []

#         # Title
#         title = Paragraph("📊 TaskPilot AI Report", styles['Title'])
#         flowables.append(title)
#         flowables.append(Spacer(1, 0.3 * inch))

#         # Insights
#         flowables.append(Paragraph("📌 Insights:", styles['Heading2']))
#         flowables.append(Paragraph(insights, styles['Normal']))
#         flowables.append(Spacer(1, 0.2 * inch))

#         # EDA Charts
#         flowables.append(Paragraph("📈 EDA Charts:", styles['Heading2']))
#         for chart in charts:
#             if os.path.exists(chart):
#                 img = Image(chart, width=5*inch, height=3*inch)
#                 flowables.append(img)
#                 flowables.append(Spacer(1, 0.2 * inch))
#             else:
#                 flowables.append(Paragraph(f"Chart not found: {chart}", styles['Normal']))

#         # Logs
#         flowables.append(Spacer(1, 0.3 * inch))
#         flowables.append(Paragraph("🧾 Logs and Metrics:", styles['Heading2']))
#         log_text = "<br />".join([f"<b>{key}</b>: {value}" for key, value in logs.items()])
#         flowables.append(Paragraph(log_text, styles['Normal']))

#         # Build PDF
#         doc.build(flowables)

#         return output_path

