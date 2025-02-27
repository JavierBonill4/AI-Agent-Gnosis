import autogen
import os
import requests
from dotenv import load_dotenv
from prediction_market_agent_tooling.deploy.agent import DeployableTraderAgent
from prediction_market_agent_tooling.markets.agent_market import AgentMarket
from prediction_market_agent_tooling.markets.data_models import ProbabilisticAnswer
from prediction_market_agent_tooling.markets.markets import MarketType
from prediction_market_agent_tooling.gtypes import Probability
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import SecretStr

# 🔹 Load API Keys from .env
load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
BET_FROM_PRIVATE_KEY = os.getenv("BET_FROM_PRIVATE_KEY")
GRAPH_API_KEY = os.getenv("GRAPH_API_KEY")
MANIFOLD_API_KEY = os.getenv("MANIFOLD_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# 🔹 AutoGen Configuration
config_list = [{"model": "gpt-4o-mini-2024-07-18", "api_key": OPENAI_API_KEY}]

# 🔹 Define AutoGen Agents
market_data_agent = autogen.AssistantAgent(name="MarketDataAgent", llm_config={"config_list": config_list, "temperature": 0.5})
news_research_agent = autogen.AssistantAgent(name="NewsResearchAgent", llm_config={"config_list": config_list, "temperature": 0.7})
web_scraper_agent = autogen.AssistantAgent(name="WebScraperAgent", llm_config={"config_list": config_list, "temperature": 0.7})
graph_data_agent = autogen.AssistantAgent(name="GraphDataAgent", llm_config={"config_list": config_list, "temperature": 0.5})
betting_agent = autogen.AssistantAgent(name="BettingAgent", llm_config={"config_list": config_list, "temperature": 0.3})

# 🔹 AI-Powered gpt-4o-mini-2024-07-18 Model
llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    api_key=SecretStr(OPENAI_API_KEY),
    temperature=0.5,
)

# 🔹 Functions for Fetching Data

def get_news_results(query: str):
    """ Fetches news using SERPER API. """
    response = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
        json={"q": query},
    )
    return [x["link"] for x in response.json().get("organic", []) if x.get("link")]

def get_market_odds(market: AgentMarket):
    """ Retrieves the current market odds from Manifold API. """
    response = requests.get(f"https://manifold.markets/api/v0/markets/{market.id}")
    if response.status_code == 200:
        market_data = response.json()
        return market_data.get("probability", 0.5)  # Default to 50% if not available
    return 0.5

def estimate_probability(question: str, market_odds: float, news_links: list[str]):
    """ Uses gpt-4o-mini-2024-07-18 to estimate probability based on news and market odds. """
    
    prompt = ChatPromptTemplate(
        [
            ("system", "You are an expert prediction market trader."),
            (
                "user",
                f"""Today is {os.getenv('CURRENT_DATE', 'unknown')}.

Given the following question: "{question}"
And the current market odds: {market_odds}
And the latest news articles: {news_links}

Estimate the probability of this event happening (0.0 to 1.0).
Return only the probability float number and confidence float number, separated by space.
                """,
            ),
        ]
    )
    
    messages = prompt.format_messages()
    probability_and_confidence = str(llm.invoke(messages, max_tokens=10).content)
    
    try:
        probability, confidence = map(float, probability_and_confidence.split())
        return probability, confidence
    except:
        print(f"❌ Failed to parse probability response: {probability_and_confidence}")
        return 0.5, 0.5  # Default to neutral estimate

# 🔹 Define the Trading Agent
class YourAgent(DeployableTraderAgent):
    bet_on_n_markets_per_run = 1  # Can be adjusted later

    def answer_binary_market(self, market: AgentMarket) -> ProbabilisticAnswer | None:
        """
        Uses AI and market data to decide whether to bet.
        """
        print(f"📊 Market Question: {market.question}")

        # Step 1: Fetch Latest Market Odds
        market_odds = get_market_odds(market)
        print(f"📈 Market Odds: {market_odds}")

        # Step 2: Get News Articles
        news_links = get_news_results(market.question)
        print(f"📰 Found {len(news_links)} news articles.")

        # Step 3: Use gpt-4o-mini-2024-07-18 to Estimate Probability
        probability, confidence = estimate_probability(market.question, market_odds, news_links)
        print(f"🤖 AI Estimated Probability: {probability}, Confidence: {confidence}")

        # Step 4: Decision Logic (Only Bet If AI Probability > Market Odds)
        if probability > market_odds:
            print(f"✅ Betting because AI thinks {probability} > {market_odds}")
            return ProbabilisticAnswer(
                confidence=confidence,
                p_yes=Probability(probability),
                reasoning="AI-based estimate using market data and news insights.",
            )
        else:
            print(f"❌ No bet placed because AI thinks {probability} <= {market_odds}")
            return None  # No bet

if __name__ == "__main__":
    agent = YourAgent()
    
    # Runs the agent on real prediction markets
    agent.run(market_type=MarketType.OMEN)
