import json

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

load_dotenv(find_dotenv(), override=True)

# Choose the right model https://openai.com/pricing
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
vector_stores = {}


class GetStockTickerTool(BaseTool):
    name = "get-stock-ticker"
    description = "Lookup stock ticker based on stock name"

    def _run(self, query: str) -> str:
        """Use the LLM to look up stock information."""
        try:
            question = f"Stock symbol of {query}"
            vector_store = get_db(category="StockInfo")
            answer = ask_and_get_answer(vector_store, question, 3)

            return answer
        except Exception as e:
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
            print(f"Query: {query}")
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
            response += f"dividendYield={info['dividendYield']}, "
            response += f"dividendRate={info['dividendRate']}, "
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

            print(response)
            return response
        except Exception as e:
            return f"Sorry I could not find the current price of {query}"

    async def _arun(self, query: str) -> str:
        """Use the yahoo-finance API to look up stock information."""
        return await self._run(query)


def load_tickers_to_db():
    chunk_size = 512
    data = DocumentLoader.load_document('uploads/stock_tickers.txt')
    chunks = Chunker.chunk_data(data, chunk_size=chunk_size)
    create_embeddings(chunks, category="StockInfo")


def create_generic_agent_chain(tools, llm, agent_type):
    return initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)


def call_finance_data_handler(user_input: str):
    # Only load if some information changed
    # load_tickers_to_db()

    tools = [
        GetStockTickerTool(),
        YahooFinanceTool()
    ]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Make sure to use the get-stock-ticker and yahoo-finance tools for information that you couldn't find, However, if the selected prompt does not offer useful information or is not applicable, simply state 'No answer found'.",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    agent_executor = AgentExecutor(agent=create_tool_calling_agent(llm, tools, prompt), tools=tools, verbose=False)
    result = agent_executor.invoke({"input": f"{user_input}"})

    # agent_chain = create_generic_agent_chain(tools=tools, llm=llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    # result = agent_chain.invoke({"input": "What is the current price of Blabna stock?"})
    print(result)

    return result


if __name__ == '__main__':
    load_tickers_to_db()
#     call_finance_data_handler(user_input="What was performance of Microsoft on 04/16 of this year?")
#     input_str = """
#     Analyze Nvidia stock and let me know if its performing well and if you think it will grow. I don't need a detailed breakdown, give me a high level summary with some key points. Use the below information for answering the question:
#
# The stock details of NVDA are shown as key value pairs: CurrentPrice=762.0, PreviousClose=846.71, DayHigh=843.24, DayLow=756.06, 52WeekChange=1.8178389, Trailing PE=63.712376, Forward PE=26.458334, PriceToBook PE=43.687653, BookValue=17.442, dividendYield=0.0002, dividendRate=0.16, payoutRatio=0.0134000005, volume=86438437, averageVolume=53010828, averageVolume10days=47033170, averageDailyVolume10Day=47033170, bidSize=100, askSize=100, beta=1.744, shortRatio=0.48, shortPercentOfFloat=0.0117999995, auditRisk=7, boardRisk=10, profitMargins=0.48849, returnOnEquity=0.91458, returnOnAssets=0.38551, DebtToEquity=25.725, currentRatio=4.171, quickRatio=3.385, TotalCash=25984000000, EBITDA=34480001024, revenueGrowth=2.653, earningsGrowth=7.613, freeCashflow=19866875904, grossMargins=0.72718.targetHighPrice=2594.96, targetLowPrice=449.45, targetMeanPrice=942.65, targetMedianPrice=939.48, recommendationMean=1.7, numberOfAnalystOpinions=47, heldPercentInsiders=0.040009998, heldPercentInstitutions=0.67985;
# The last 5 day performance is as follows:                                  Open        High         Low       Close    Volume  Dividends  Stock Splits
# Date
# 2024-04-15 00:00:00-04:00  890.979980  906.130005  859.289978  860.010010  44307700        0.0           0.0
# 2024-04-16 00:00:00-04:00  864.330017  881.179993  860.640015  874.150024  37045300        0.0           0.0
# 2024-04-17 00:00:00-04:00  883.400024  887.750000  839.500000  840.349976  49540000        0.0           0.0
# 2024-04-18 00:00:00-04:00  849.700012  861.900024  824.020020  846.710022  44726000        0.0           0.0
# 2024-04-19 00:00:00-04:00  831.500000  843.239990  756.059998  762.000000  87190500        0.0           0.0
#     """
