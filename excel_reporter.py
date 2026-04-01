import openpyxl
import os
from datetime import datetime

EXCEL_FILE = "לקוחות_רועי.xlsx"

def init_excel():
    if not os.path.exists(EXCEL_FILE):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "לקוחות"
        headers = ["תאריך", "שם", "אזור", "חדרים", "תקציב", "סוג", "סטטוס", "הערות"]
        ws.append(headers)
        for col in range(1, 9):
            ws.column_dimensions[openpyxl.utils.get_column_letter(col)].width = 18
        wb.save(EXCEL_FILE)
        print(f"קובץ אקסל נוצר: {EXCEL_FILE}")

def add_client(name: str, area: str, rooms: str, budget: str, deal_type: str = "לא ידוע", notes: str = ""):
    init_excel()
    wb = openpyxl.load_workbook(EXCEL_FILE)
    ws = wb.active
    
    row = [
        datetime.now().strftime("%d/%m/%Y %H:%M"),
        name,
        area,
        rooms,
        budget,
        deal_type,
        "חדש",
        notes
    ]
    
    ws.append(row)
    wb.save(EXCEL_FILE)
    print(f"לקוח נוסף לאקסל: {name}")