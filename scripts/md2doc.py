#!/usr/bin/env python3
"""
将 Markdown 转换为 Word 文档
"""

import markdown
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
import sys
import re

def create_doc_from_markdown(md_file, doc_file):
    # 读取Markdown文件
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # 创建Word文档
    doc = Document()
    
    # 设置默认字体
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Calibri'
    font.size = Pt(11)
    
    # 解析Markdown
    lines = md_content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # 跳过空行
        if not line.strip():
            i += 1
            continue
        
        # 标题处理
        if line.startswith('# '):
            # 文档标题
            p = doc.add_heading(line[2:], level=0)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif line.startswith('## '):
            p = doc.add_heading(line[3:], level=1)
        elif line.startswith('### '):
            p = doc.add_heading(line[4:], level=2)
        elif line.startswith('#### '):
            p = doc.add_heading(line[5:], level=3)
        
        # 分隔线
        elif line.strip() == '---':
            pass  # Word中不需要单独处理
        
        # 表格
        elif line.startswith('|') and i + 1 < len(lines) and lines[i+1].startswith('|---'):
            # 开始表格
            table_lines = []
            j = i
            while j < len(lines) and lines[j].strip():
                if lines[j].startswith('|'):
                    table_lines.append(lines[j])
                else:
                    break
                j += 1
            
            if table_lines:
                # 创建表格
                headers = [h.strip() for h in table_lines[0].split('|')[1:-1]]
                table = doc.add_table(rows=len(table_lines)-1, cols=len(headers))
                table.style = 'Table Grid'
                
                # 填充表头
                for k, header in enumerate(headers):
                    table.rows[0].cells[k].text = header
                
                # 填充数据行
                for row_idx, row_line in enumerate(table_lines[2:]):
                    cells = [c.strip() for c in row_line.split('|')[1:-1]]
                    for k, cell_text in enumerate(cells):
                        table.rows[row_idx+1].cells[k].text = cell_text
            
            i = j
            continue
        
        # 列表项
        elif line.startswith('- ') or line.startswith('* '):
            p = doc.add_paragraph()
            p.add_run(line[2:])
        
        # 代码块
        elif line.startswith('```'):
            # 收集代码块内容
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code_lines.append(lines[i])
                i += 1
            
            p = doc.add_paragraph()
            code_run = p.add_run('\n'.join(code_lines))
            code_run.font.name = 'Consolas'
            code_run.font.size = Pt(10)
        
        # 普通段落
        else:
            # 处理行内格式
            paragraph_text = line
            
            # 处理加粗 **text**
            bold_pattern = r'\*\*([^*]+)\*\*'
            # 处理行内代码 `code`
            code_pattern = r'`([^`]+)`'
            
            p = doc.add_paragraph()
            
            # 分割文本处理不同格式
            parts = re.split(r'(\*\*|\`)', paragraph_text)
            is_bold = False
            is_code = False
            
            for part in parts:
                if part == '**':
                    is_bold = not is_bold
                    continue
                elif part == '`':
                    is_code = not is_code
                    continue
                
                if part:
                    run = p.add_run(part)
                    if is_bold:
                        run.bold = True
                    if is_code:
                        run.font.name = 'Consolas'
                        run.font.size = Pt(10)
        
        i += 1
    
    # 添加分隔页
    doc.add_page_break()
    
    # 添加版权信息
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('\n\n© 2026 Tira 炒股小帮手\n基于 OpenClaw + 美股回测系统构建')
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(128, 128, 128)
    
    # 保存文档
    doc.save(doc_file)
    print(f"已生成: {doc_file}")

if __name__ == '__main__':
    md_file = 'docs/README_TIRA.md'
    doc_file = 'docs/README_TIRA.docx'
    create_doc_from_markdown(md_file, doc_file)