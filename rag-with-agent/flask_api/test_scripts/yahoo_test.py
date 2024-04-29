import json

from langchain.tools.base import BaseTool
import yfinance as yf
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

# Choose the right model https://openai.com/pricing
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
vector_stores = {}


def test_yahoo_finance(stock_symbol):
    try:
        print(f"Query: {stock_symbol}")
        ticker = stock_symbol.upper()
        stock = yf.Ticker(ticker)
        info = stock.info
        print(json.dumps(info, indent=4))

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

        print(response)
        return response
    except Exception as e:
        print(e)
        print(f"Sorry I could not find the current price of {stock_symbol}")
        return f"Sorry I could not find the current price of {stock_symbol}"


# def load_tickers_to_db():
#     chunk_size = 512
#     data = DocumentLoader.load_document('uploads/stock_tickers.txt')
#     chunks = Chunker.chunk_data(data, chunk_size=chunk_size)
#     create_embeddings(chunks, category="StockInfo")

if __name__ == '__main__':
    # load_tickers_to_db()

    test_yahoo_finance(stock_symbol="NFLX")
