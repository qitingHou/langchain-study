from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate  #ChatPromptTemplate用来创建与 LLM 交互的提示模板。
from langchain_core.runnables import RunnablePassthrough  #RunnablePassthrough用于简单的无修改传递数据。
import getpass
import os
import langchain
from langchain.schema.messages import HumanMessage, SystemMessage,AIMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
client = OpenAI(
    api_key="OPENAI-API-KEY",
)
model = ChatOpenAI(model="gpt-3.5-turbo")

documents = [    #生成几个文件
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",#包含了关于某个宠物的信息。
        metadata={"source": "mammal-pets-doc"},#包含该文档的元数据，在这里显示的是该文档的来源（source）。
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

#向量存储
vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)
vectorstore.similarity_search_with_score("cat")
embedding = OpenAIEmbeddings().embed_query("cat")
vectorstore.similarity_search_by_vector(embedding)

#检索器
retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)  # select top result
retriever.batch(["cat", "shark"])

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)
retriever.batch(["cat", "shark"])

llm = ChatOpenAI(model="gpt-3.5-turbo")
message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""
prompt = ChatPromptTemplate.from_messages([("human", message)])
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

response = rag_chain.invoke("tell me about dog")
print(response.content)