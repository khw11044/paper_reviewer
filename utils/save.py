import os
from bs4 import BeautifulSoup
from markdownify import markdownify as markdown

def save_results(output_folder, filename, html_content):
    
    # 1. HTML 파일 저장
    html_output_file = os.path.join(output_folder, f"{filename}.html")
    combined_html_content = "\n".join(html_content)
    soup = BeautifulSoup(combined_html_content, "html.parser")
    all_tags = set([tag.name for tag in soup.find_all()])
    html_tag_list = [tag for tag in list(all_tags) if tag not in ["br"]]

    with open(html_output_file, "w", encoding="utf-8") as f:
        f.write(combined_html_content)
    
    print(f"HTML 파일이 {html_output_file}에 저장되었습니다.")
    
    # 2. Markdown 파일 저장
    md_output_file = os.path.join(output_folder, f"{filename}.md")
    md_output = markdown(
        combined_html_content,
        convert=html_tag_list,
    )

    with open(md_output_file, "w", encoding="utf-8") as f:
        f.write(md_output)

    print(f"Markdown 파일이 {md_output_file}에 저장되었습니다.")
    
    return md_output_file

