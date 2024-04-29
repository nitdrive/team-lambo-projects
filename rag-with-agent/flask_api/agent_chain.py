import json
from typing import Optional

from langchain.tools.base import BaseTool
import yfinance as yf
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from services.loaders import DocumentLoader
from services.chunkers import Chunker
from embeddings import create_embeddings, ask_and_get_answer, get_db
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
import requests, json
import os

load_dotenv(find_dotenv(), override=True)

# Choose the right model https://openai.com/pricing
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
vector_stores = {}


class GetStockTickerTool(BaseTool):
    name = "get-stock-ticker"
    description = "Lookup stock ticker based on stock name. If stock symbol is not found then skip this tool"

    def _run(self, query: str) -> str:
        """Use the LLM to look up stock information."""
        try:
            print("Querying GetStockTickerTool")
            question = f"Stock symbol of {query}"
            print(question)
            vector_store = get_db(category="StockInfo")
            answer = ask_and_get_answer(vector_store, question, 3)
            print(answer)
            return answer
        except Exception as e:
            print("Exception in GetStockTickerTool")
            print(e)
            raise f"Error: {e}"
            # return f"Error: {e}"

    async def _arun(self, query: str) -> str:
        """Use the yahoo-finance API to look up stock information."""
        return await self._run(query)


class YahooFinanceTool(BaseTool):
    name = "yahoo-finance"
    description = "Lookup stock information from Yahoo Finance. Note: you need to convert the stock name to a ticker " \
                  "before using this API "

    def _run(self, query: str) -> str:
        """Use the yahoo-finance API to look up stock information."""
        try:
            print(f"Querying YahooFinanceTool: {query}")
            ticker = query.upper()
            stock = yf.Ticker(ticker)
            info = stock.info
            # print(json.dumps(info, indent=4))

            # Include any relevant stock information
            response = f"The stock details of {ticker} are shown below as key value pairs: "

            # Market performance metrics
            response += f"CurrentPrice={info['currentPrice']}, "
            response += f"PreviousClose={info['previousClose']}, "
            response += f"DayHigh={info['dayHigh']}, "
            response += f"DayLow={info['dayLow']}, "
            response += f"52WeekChange={info['52WeekChange']}, "

            # Valuation metrics
            response += f"Trailing PE={info['trailingPE']}, "
            response += f"Forward PE={info['forwardPE']}, "
            response += f"PriceToBook PE={info['priceToBook']}, "
            response += f"BookValue={info['bookValue']}, "

            # Dividend Metrics
            if 'dividendYield' in info:
                response += f"dividendYield={info['dividendYield']}, "
            if 'dividendRate' in info:
                response += f"dividendRate={info['dividendRate']}, "
            if 'payoutRatio' in info:
                response += f"payoutRatio={info['payoutRatio']}, "

            # Volume and Liquidity Metrics
            response += f"volume={info['volume']}, "
            response += f"averageVolume={info['averageVolume']}, "
            response += f"averageVolume10days={info['averageVolume10days']}, "
            response += f"averageDailyVolume10Day={info['averageDailyVolume10Day']}, "
            response += f"bidSize={info['bidSize']}, "
            response += f"askSize={info['askSize']}, "

            # Risk Assessment
            response += f"beta={info['beta']}, "
            response += f"shortRatio={info['shortRatio']}, "
            response += f"shortPercentOfFloat={info['shortPercentOfFloat']}, "
            response += f"auditRisk={info['auditRisk']}, "
            response += f"boardRisk={info['boardRisk']}, "

            # Profitability and Efficiency
            response += f"profitMargins={info['profitMargins']}, "
            response += f"returnOnEquity={info['returnOnEquity']}, "
            response += f"returnOnAssets={info['returnOnAssets']}, "

            # Financial Health
            response += f"DebtToEquity={info['debtToEquity']}, "
            response += f"currentRatio={info['currentRatio']}, "
            response += f"quickRatio={info['quickRatio']}, "
            response += f"TotalCash={info['totalCash']}, "
            response += f"EBITDA={info['ebitda']}, "

            # Growth Indicators
            response += f"revenueGrowth={info['revenueGrowth']}, "
            response += f"earningsGrowth={info['earningsGrowth']}, "
            response += f"freeCashflow={info['freeCashflow']}, "
            response += f"grossMargins={info['grossMargins']}."

            # Analyst Opinions and Future Projections
            response += f"targetHighPrice={info['targetHighPrice']}, "
            response += f"targetLowPrice={info['targetLowPrice']}, "
            response += f"targetMeanPrice={info['targetMeanPrice']}, "
            response += f"targetMedianPrice={info['targetMedianPrice']}, "
            response += f"recommendationMean={info['recommendationMean']}, "
            response += f"numberOfAnalystOpinions={info['numberOfAnalystOpinions']}, "

            # Governance and Institutional Interest
            response += f"heldPercentInsiders={info['heldPercentInsiders']}, "
            response += f"heldPercentInstitutions={info['heldPercentInstitutions']}, "

            response += "\n"

            # Include any relevant historic performance information
            history = stock.history(period="5d")

            response += f"The last 5 day performance is as follows: {history.to_string()}"

            return response
        except Exception as e:
            print(e)
            return f"Sorry I could not find the current price of {query}"

    async def _arun(self, query: str) -> str:
        """Use the yahoo-finance API to look up stock information."""
        return await self._run(query)


class MambaQueryTool(BaseTool):
    name = "general-mamba-query-tool"
    description = "Lookup general queries"

    def _run(self, query: str) -> str:
        """Use the mamba LLM to look up db information."""
        print(f"Querying MambaQueryTool: {query}")
        headers = {"Authorization": f"Bearer {os.environ.get('HF_API_TOKEN')}", "Content-Type": "application/json"}
        try:
            data = {
                "inputs": {
                    "query": f"{query}"
                }
            }
            response = requests.post(url=os.environ.get('MAMBA_API_URL'), headers=headers, data=json.dumps(data))

            result = response.content.decode('utf-8')
            print(result)
            result = result.split('<|assistant|>')[1].replace("<|endoftext|>\"}", "")
            print(f"Processed: {result}")

            return result
        except Exception as e:
            print("Error MambaQueryTool")
            print(e)
            raise f"Error: {e}"

    async def _arun(self, query: str) -> str:
        return await self._run(query)


class GeneralQueryTool(BaseTool):
    name = "general-query-tool"
    description = "Lookup general queries"

    def _run(self, query: str) -> str:
        """Use the LLM to look up db information."""
        try:
            print(f"Querying GeneralQueryTool: {query}")
            messages = [
                ("system", "You are a helpful assistant that answers questions"),
                ("human", f"{query}."),
            ]

            answer = llm.invoke(messages)
            print(answer.content)
            return answer.content
        except Exception as e:
            print("Error GeneralQueryTool")
            print(e)
            raise f"Error: {e}"
            # return f"Error: {e}"

    async def _arun(self, query: str) -> str:
        return await self._run(query)


class VanguardQueryTool(BaseTool):
    name = "vanguard-query-tool"
    description = "Lookup queries about Vanguard"

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Use the LLM to look up Vanguard information."""
        try:
            print(f"Querying VanguardQueryTool: {query}")
            vector_store = get_db()
            answer = ask_and_get_answer(vector_store, query, 3, called_from='VanguardQueryTool')
            print(answer)
            return answer
        except Exception as e:
            print("Error VanguardQueryTool")
            print(e)
            raise f"Error: {e}"
            # return f"Error: {e}"

    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        return await self._run(query)


def execute_chain(user_input: str):
    # GeneralQueryTool()
    tools = [
        VanguardQueryTool(),
        GetStockTickerTool(),
        YahooFinanceTool(),
        # MambaQueryTool(),
        GeneralQueryTool()
    ]

    # "You are a helpful assistant. Make sure to use the get-stock-ticker and yahoo-finance and general-query-tool for finding information. And vanguard-query-tool for finding information about Vanguard. However, if the selected prompt does not offer useful information or is not applicable, simply state 'No answer found'.",

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that answers questions. If a question has derogatory or sexual references don't try to answer it. Note: Not all companies will have stock symbol so if you cannot find something using get-stock-ticker tool, then use the general-mamba-query-tool tool. Show the response as is don't summarize it.",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("assistant", "{chat_history}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             "You are a helpful assistant. Make sure to use the get-stock-ticker and yahoo-finance and vanguard-query-tool for finding information. However, if the selected prompt does not offer useful information or is not applicable, simply state 'No answer found'.",
    #         ),
    #         ("placeholder", "{chat_history}"),
    #         ("human", "{input}"),
    #         ("placeholder", "{agent_scratchpad}"),
    #     ]
    # )

    agent_executor = AgentExecutor(agent=create_tool_calling_agent(llm, tools, prompt), tools=tools, verbose=False)
    result = agent_executor.invoke({"input": f"{user_input}"}, tags=[user_input])

    print(result)

    return result
