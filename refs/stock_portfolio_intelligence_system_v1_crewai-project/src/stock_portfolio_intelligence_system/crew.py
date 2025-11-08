import os

from crewai import LLM
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import (
	ScrapeWebsiteTool,
	SerplyNewsSearchTool,
	BraveSearchTool
)
from stock_portfolio_intelligence_system.tools.telegram_bot_messenger import TelegramBotMessenger




@CrewBase
class StockPortfolioIntelligenceSystemCrew:
    """StockPortfolioIntelligenceSystem crew"""

    
    @agent
    def market_data_collector(self) -> Agent:
        
        return Agent(
            config=self.agents_config["market_data_collector"],
            
            
            tools=[				ScrapeWebsiteTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def news_sentiment_analyst(self) -> Agent:
        
        return Agent(
            config=self.agents_config["news_sentiment_analyst"],
            
            
            tools=[				SerplyNewsSearchTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def sec_filings_analyst(self) -> Agent:
        
        return Agent(
            config=self.agents_config["sec_filings_analyst"],
            
            
            tools=[				ScrapeWebsiteTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def technical_analysis_expert(self) -> Agent:
        
        return Agent(
            config=self.agents_config["technical_analysis_expert"],
            
            
            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def portfolio_strategy_manager(self) -> Agent:
        
        return Agent(
            config=self.agents_config["portfolio_strategy_manager"],
            
            
            tools=[],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def advanced_investment_researcher(self) -> Agent:
        
        return Agent(
            config=self.agents_config["advanced_investment_researcher"],
            
            
            tools=[				BraveSearchTool(),
				SerplyNewsSearchTool(),
				ScrapeWebsiteTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def social_media_intelligence_analyst(self) -> Agent:
        
        return Agent(
            config=self.agents_config["social_media_intelligence_analyst"],
            
            
            tools=[				BraveSearchTool(),
				ScrapeWebsiteTool()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    
    @agent
    def communication_manager(self) -> Agent:
        
        return Agent(
            config=self.agents_config["communication_manager"],
            
            
            tools=[				TelegramBotMessenger()],
            reasoning=False,
            max_reasoning_attempts=None,
            inject_date=True,
            allow_delegation=False,
            max_iter=25,
            max_rpm=None,
            
            max_execution_time=None,
            llm=LLM(
                model="gpt-4o-mini",
                temperature=0.7,
            ),
            
        )
    

    
    @task
    def collect_market_data(self) -> Task:
        return Task(
            config=self.tasks_config["collect_market_data"],
            markdown=False,
            
            
        )
    
    @task
    def analyze_news_and_sentiment(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_news_and_sentiment"],
            markdown=False,
            
            
        )
    
    @task
    def review_sec_filings(self) -> Task:
        return Task(
            config=self.tasks_config["review_sec_filings"],
            markdown=False,
            
            
        )
    
    @task
    def social_media_stock_discovery(self) -> Task:
        return Task(
            config=self.tasks_config["social_media_stock_discovery"],
            markdown=False,
            
            
        )
    
    @task
    def perform_technical_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["perform_technical_analysis"],
            markdown=False,
            
            
        )
    
    @task
    def advanced_investment_decision_analysis(self) -> Task:
        return Task(
            config=self.tasks_config["advanced_investment_decision_analysis"],
            markdown=False,
            
            
        )
    
    @task
    def send_telegram_alerts(self) -> Task:
        return Task(
            config=self.tasks_config["send_telegram_alerts"],
            markdown=False,
            
            
        )
    
    @task
    def final_portfolio_strategy_report(self) -> Task:
        return Task(
            config=self.tasks_config["final_portfolio_strategy_report"],
            markdown=False,
            
            
        )
    
    @task
    def send_investment_alerts(self) -> Task:
        return Task(
            config=self.tasks_config["send_investment_alerts"],
            markdown=False,
            
            
        )
    

    @crew
    def crew(self) -> Crew:
        """Creates the StockPortfolioIntelligenceSystem crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )

    def _load_response_format(self, name):
        with open(os.path.join(self.base_directory, "config", f"{name}.json")) as f:
            json_schema = json.loads(f.read())

        return SchemaConverter.build(json_schema)
