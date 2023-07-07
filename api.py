from typing import Any, Type
from langchain.llms import OpenAI
from langchain.agents import AgentType, ConversationalAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, BaseSettings, Field
from langchain.tools import BaseTool
from langchain.tools.playwright.utils import create_async_playwright_browser
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from gradio_tools.tools import StableDiffusionTool
import os

os.environ["OPENAI_API_KEY"] = "your_api_key"


class PlaywrightBrowserSchema(BaseModel):
    path: str = Field(default="", description="the api path")


class PlaywrightBrowser(BaseTool, BaseSettings):
    name: str = "navigate_stateless_browser"
    description: str = """Tool to navigate the web if the website is dynamically rendered."""
    args_schema: Type[PlaywrightBrowserSchema] = PlaywrightBrowserSchema

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        async_browser = create_async_playwright_browser()
        browser_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
        return browser_toolkit.get_tools()

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        pass


class ReekAgent:
    def __init__(self, llm, tools, memory) -> None:
        self.prefix = "The following is a conversation between you and a user."
        self.ai_prefix = "Reek"
        self.human_prefix = "User"
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.agent = self.create_agent()

    def create_agent(self):
        reek_agent = ConversationalAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            prefix=self.prefix,
            ai_prefix=self.ai_prefix,
            human_prefix=self.human_prefix,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
        )
        agent_executor = AgentExecutor.from_agent_and_tools(agent=reek_agent,
                                                            tools=self.tools,
                                                            verbose=True,
                                                            memory=self.memory)
        return agent_executor

    def run(self, input):
        return self.agent.run(input=input)




def get_response(prompt):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    tools = [PlaywrightBrowser(), StableDiffusionTool().langchain]
    llm = OpenAI(temperature=0)
    agent = ReekAgent(llm, tools, memory)
    response = agent.run(prompt)
    return response
