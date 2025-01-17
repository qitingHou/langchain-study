import getpass
import os
import langchain
from langchain_core.messages import SystemMessage, trim_messages
from langchain.schema.messages import HumanMessage, SystemMessage,AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, trim_messages

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("OPENAI-API-KEY")

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")

parser = StrOutputParser()
from langchain_core.messages import HumanMessage
result=model.invoke([HumanMessage(content="My name is kitty")])
parser.invoke(result) #直接使用模型

from langchain_core.messages import AIMessage
result=model.invoke(
    [
        HumanMessage(content="Hi! I'm kitty"),
        AIMessage(content="Hello kitty! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)
parser.invoke(result)#直接关联上下文

#导入相关类并设置链
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
with_message_history = RunnableWithMessageHistory(model, get_session_history)
config = {"configurable": {"session_id": "aaa"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm kitty")],
    config=config,
)
response.content
response = with_message_history.invoke(        #提问
    [HumanMessage(content="What's my name?")],
    config=config,
)
response.content

config = {"configurable": {"session_id": "bbb"}}
response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)
response.content#更改配置设置不同的session_id开始新的对话

#提示词模板
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)
chain = prompt | model
response = chain.invoke({"messages": [HumanMessage(content="hi! I'm kitty")]})
response.content

#将他封装与之前相同的历史消息对象中
with_message_history = RunnableWithMessageHistory(chain, get_session_history)
config = {"configurable": {"session_id": "ccc"}}
response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Tom")],
    config=config,
)
response.content

#管理历史对话

from typing import List
# pip install tiktoken
import tiktoken
from langchain_core.messages import BaseMessage, ToolMessage
def str_token_counter(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def tiktoken_counter(messages: List[BaseMessage]) -> int:
    """Approximately reproduce https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

    For simplicity only supports str Message.contents.
    """
    num_tokens = 3  # every reply is primed with <|start|>assistant<|message|>
    tokens_per_message = 3
    tokens_per_name = 1
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported messages type {msg.__class__}")
        num_tokens += (
            tokens_per_message
            + str_token_counter(role)
            + str_token_counter(msg.content)
        )
        if msg.name:
            num_tokens += tokens_per_name + str_token_counter(msg.name)
    return num_tokens


trim_messages(
    max_tokens=45,
    strategy="last",
    token_counter=tiktoken_counter,
)

from langchain_core.messages import SystemMessage, trim_messages
trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=tiktoken_counter,
    include_system=True,
    allow_partial=False,
    start_on="human",
)
messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]
trimmer.invoke(messages)
#在链中使用
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
chain = (
    RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
    | prompt
    | model
)
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what's my name?")],
        "language": "Chinese",
    }
)
response.content
#询问最近几条消息
response = chain.invoke(
    {
        "messages": messages + [HumanMessage(content="what math problem did i ask")],
        "language": "Eglish",
    }
)
response.content
#封装历史信息
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
)
config = {"configurable": {"session_id": "fff"}}

response = with_message_history.invoke(
    {
        "messages": messages + [HumanMessage(content="whats my name?")],
        "language": "English",
    },
    config=config,
)
response.content

#流式处理
config = {"configurable": {"session_id": "abc15"}}
for r in with_message_history.stream(
    {
        "messages": [HumanMessage(content="hi! I'm kitty. tell me a new")],
        "language": "English",
    },
    config=config,
):
    print(r.content, end="|")