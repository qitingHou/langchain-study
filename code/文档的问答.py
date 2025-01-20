from langchain.chains import RetrievalQA #检索QA链，在文档上进行检索
from langchain.chat_models import ChatOpenAI #openai模型
from langchain.document_loaders import CSVLoader #文档加载器，采用csv格式存储
from langchain.vectorstores import DocArrayInMemorySearch #向量存储
from IPython.display import display, Markdown #在jupyter显示信息的工具

file_path = r"D:\langchainstudy\LangChainStudy\robot\OutdoorClothingCatalog_1000.csv"
loader = CSVLoader(file_path=file_path, encoding="utf-8")
#查看数据
import pandas as pd
data = pd.read_csv(file_path,header=None)

from langchain.indexes import VectorstoreIndexCreator  # 导入向量存储索引创建器
from langchain.embeddings.openai import OpenAIEmbeddings  # 导入嵌入模型
from langchain_openai import OpenAI  # 正确导入 OpenAI
import openai
openai.api_key = "OPENAI-API-KEY"
# 初始化嵌入模型（这里以 OpenAIEmbeddings 为例）
embedding = OpenAIEmbeddings()
llm = OpenAI(temperature=0)

# 创建索引（传入 embedding 和 vectorstore_cls）
index = VectorstoreIndexCreator(
    embedding=embedding,  # 传入嵌入模型
    vectorstore_cls=DocArrayInMemorySearch  # 传入向量存储类
).from_loaders([loader])  # 使用加载器加载文档

# 执行查询
query = "Please list all your shirts with sun protection in a table in markdown and summarize each one."
response = index.query(query,llm=llm)  # 使用索引查询创建一个响应，并传入查询

# 显示响应内容
display(Markdown(response))  # 以 Markdown 格式查看查询返回的内容