import getpass
import os
import langchain
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()
from langchain.schema.messages import HumanMessage, SystemMessage,AIMessage
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    azure_endpoint="https://12205-m2hl4tqk-eastus.openai.azure.com/openai/deployments/gpt-35-turbo/chat/completions?api-version=2024-08-01-preview",
    azure_deployment="gpt-35-turbo",
    openai_api_version="2024-08-01-preview",
    api_key="d7f27353b3b3463bb02b2708df922f35",  
)#密钥
messages = [
    SystemMessage(content="用中文回复"),
    HumanMessage(content="中国有几个省份"),
  
]
result = model.invoke(messages)
parser.invoke(result)#只输出想要得到的字符串相应


#提示词模板
from langchain_core.prompts import ChatPromptTemplate
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
result = prompt_template.invoke({"language": "eglish", "text": "hi"})
result#返回的是一个ChatPromptValue，由两个消息组成

result.to_messages()#直接访问消息
[SystemMessage(content='Translate the following into english:'), HumanMessage(content='hi')]
#使用LCEL组件
chain = prompt_template | model | parser
chain.invoke({"language": "eglish", "text": "你好"})