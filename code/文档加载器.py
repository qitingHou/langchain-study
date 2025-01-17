from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PythonLoader
from langchain.document_loaders import JSONLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import BSHTMLLoader
import json
from pathlib import Path
from pprint import pprint

# 加载 Text 文件
loader = TextLoader(r"documentstore\index.md")  # 注意 Windows 路径分隔符
documents = loader.load()
print(documents)

# 加载 CSV 数据，每个文档一行
loader = CSVLoader(file_path=r'documentstore\index.csv')  # 同样修改路径
data = loader.load()
print(data)

# 自定义 CSV 解析和加载
loader = CSVLoader(file_path=r'documentstore\index.csv', csv_args={
    'delimiter': ',',
    'quotechar': '"',
    'fieldnames': ['title','context']
})
data = loader.load()
print(data)

# 使用 source_column 参数指定源
loader = CSVLoader(file_path=r'documentstore\index.csv', source_column="context")
data = loader.load()
print(data)

# 从文件夹加载所有文档
loader = DirectoryLoader(r'D:\langchainstudy\langchain-robot\documentstore', glob='**/*.md')
docs = loader.load()
print(len(docs))

# 显示进度条
loader = DirectoryLoader(r'D:\langchainstudy\langchain-robot\documentstore', glob="**/*.md", show_progress=True)
docs = loader.load()

# 使用多线程
loader = DirectoryLoader(r'D:\langchainstudy\langchain-robot\documentstore', glob="**/*.md", use_multithreading=True)
docs = loader.load()
print(len(docs))

# 更改加载程序类
loader = DirectoryLoader(r'D:\langchainstudy\langchain-robot', glob="*.py", loader_cls=PythonLoader)
docs = loader.load()
print(len(docs))

# 加载 HTML 文件
loader = UnstructuredHTMLLoader(r"documentstore\fake-content.html")  
data = loader.load()
print(data)

# 使用 BeautifulSoup4 加载 HTML 文件
loader = BSHTMLLoader(r"documentstore\fake-content.html")
data = loader.load()
print(data)

# 处理 JSON 文件
file_path = r'./documentstore\examples.json'  
data = json.loads(Path(file_path).read_text())
pprint(data)
