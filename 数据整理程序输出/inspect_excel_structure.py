# -*- coding: utf-8 -*-
"""Inspect .xlsx sheet names and headers without third-party dependencies."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "数据"
OUT_DIR = Path(__file__).resolve().parent
REPORT_PATH = OUT_DIR / "00_excel_structure_report.md"

NS = {
	"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
	"rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def col_to_index(cell_ref: str) -> int:
	letters = re.sub(r"[^A-Z]", "", cell_ref.upper())
	idx =0
	for ch in letters:
		idx = idx *26 + ord(ch) - ord("A") +1
	return idx -1


def read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
	try:
		data = zf.read("xl/sharedStrings.xml")
	except KeyError:
		return []
	root = ET.fromstring(data)
	strings = []
	for si in root.findall("main:si", NS):
		texts = [t.text or "" for t in si.findall(".//main:t", NS)]
		strings.append("".join(texts))
	return strings


def workbook_sheets(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
	wb = ET.fromstring(zf.read("xl/workbook.xml"))
	rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
	rel_map = {
		rel.attrib["Id"]: rel.attrib["Target"]
		for rel in rels.findall("rel:Relationship", NS)
	}
	sheets = []
	rid_key = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
	for sheet in wb.findall("main:sheets/main:sheet", NS):
		name = sheet.attrib["name"]
		target = rel_map[sheet.attrib[rid_key]]
		path = "xl/" + target.lstrip("/")
		if path.startswith("xl/../"):
			path = path.replace("xl/../", "",1)
		sheets.append((name, path))
	return sheets


def cell_value(cell: ET.Element, shared: list[str]) -> str:
	typ = cell.attrib.get("t")
	if typ == "inlineStr":
		texts = [t.text or "" for t in cell.findall(".//main:t", NS)]
		return "".join(texts)
	v = cell.find("main:v", NS)
	if v is None or v.text is None:
		return ""
	raw = v.text
	if typ == "s":
		try:
			return shared[int(raw)]
		except Exception:
			return raw
	return raw


def read_rows(zf: zipfile.ZipFile, sheet_path: str, shared: list[str], max_rows: int =5) -> list[list[str]]:
	root = ET.fromstring(zf.read(sheet_path))
	rows = []
	for row in root.findall("main:sheetData/main:row", NS):
		max_col = -1
		cells = []
		for cell in row.findall("main:c", NS):
			idx = col_to_index(cell.attrib.get("r", "A1"))
			max_col = max(max_col, idx)
			cells.append((idx, cell_value(cell, shared)))
		if max_col >=0:
			values = [""] * (max_col +1)
			for idx, value in cells:
				values[idx] = value
			rows.append(values)
		if len(rows) >= max_rows:
			break
	return rows


def main() -> None:
	lines = ["# Excel 数据结构检查报告", ""]
	for path in sorted(DATA_DIR.glob("*.xlsx")):
		lines.append(f"## {path.name}")
		try:
			with zipfile.ZipFile(path) as zf:
				shared = read_shared_strings(zf)
				sheets = workbook_sheets(zf)
				lines.append(f"- 工作表：{', '.join(name for name, _ in sheets)}")
				for sheet_name, sheet_path in sheets:
					rows = read_rows(zf, sheet_path, shared, max_rows=4)
					lines.append(f"### Sheet: {sheet_name}")
					if not rows:
						lines.append("空表")
						continue
					lines.append("字段：" + " | ".join(str(x) for x in rows[0]))
					for r in rows[1:3]:
						lines.append("样例：" + " | ".join(str(x) for x in r))
		except Exception as exc:
			lines.append(f"读取失败：{exc!r}")
		lines.append("")
	REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
	print(REPORT_PATH)


if __name__ == "__main__":
	main()
