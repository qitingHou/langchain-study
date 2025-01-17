import getpass
import os
import langchain
from langchain.schema.messages import HumanMessage, SystemMessage,AIMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults #Tavily 是一个用于语义搜索的工具，可以根据查询返回相关的搜索结果。
from openai import OpenAI
from langchain_core.messages import HumanMessage
from langchain.tools import Tool
from langgraph.prebuilt import create_react_agent 

from langgraph.checkpoint.memory import MemorySaver

os.environ["TAVILY_API_KEY"] = "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"

search = TavilySearchResults(max_results=1)#创建一个 TavilySearchResults 实例 search，设置每次搜索时最多返回 1 个结果。max_results 参数控制返回结果的数量，以避免过多的返回项。
search_results = search.invoke("what is the weather in Chongqing")
print(search_results)

#使用语言模型
api_key="OPENAI-API-KEY",
model = ChatOpenAI(model="gpt-3.5-turbo")

response = model.invoke([HumanMessage(content="hi!")])
response.content

tools = [
    Tool(
        name="WeatherTool",  # 工具名称应符合规则：字母、数字、下划线或短横线
        func=lambda location: f"The weather in {location} is sunny.",  # 工具功能
        description="Returns weather information for a given location."
    ),
    Tool(
        name="MathTool",
        func=lambda x: str(eval(x)),  # 进行简单的数学运算
        description="Performs basic math operations."
    )
]

# 将工具绑定到模型
model_with_tools = model.bind_tools(tools)

response = model_with_tools.invoke([HumanMessage(content="What is AI?")])
print(f"ContentString: {response.content}") #打印模型的回应内容
print(f"ToolCalls: {response.tool_calls}") #显示模型是否使用了外部工具，以及调用工具的详细信息

agent_executor = create_react_agent(model, tools)#创建一个反应式代理 React Agent 的函数
response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]}) #
response["messages"]

response = agent_executor.invoke(
    {"messages": [HumanMessage(content="whats the weather in Chongqing?")]}
)
response["messages"]

#流式消息
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats the weather in sf?")]}
):
    print(chunk)
    print("----")

#添加内存
memory = MemorySaver()
agent_executor = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="hi im baby!")]}, config
):
    print(chunk)
    print("----")
    
for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="whats my name?")]}, config
):
    print(chunk)
    print("----")
