import re
import pandas as pd


def extract_peptides_from_fasta_file(file_path):
    peptides = []

    with open(file_path, 'r') as file:
        fasta_content = file.read()

    # 使用正则表达式匹配FASTA格式中的肽段
    peptide_pattern = re.compile(r'>\w+\n([A-Za-z]+)')
    peptides = peptide_pattern.findall(fasta_content)

    return peptides


# 你的FASTA文件的路径
fasta_file_path = 'D:\Desktop\AMP-EF-main\YADAMP - test.fasta'

# 提取肽段
peptides = extract_peptides_from_fasta_file(fasta_file_path)

# 创建一个 pandas DataFrame
df = pd.DataFrame({'Peptide': peptides})

# 导出到 Excel 文件
excel_output_path = 'D:\Desktop\AMP-EF-main\YADAMP - test.xlsx'
df.to_excel(excel_output_path, index=False)

print(f"Peptides exported to {excel_output_path}")

