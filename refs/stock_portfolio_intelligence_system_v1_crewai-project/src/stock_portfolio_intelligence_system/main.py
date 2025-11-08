#!/usr/bin/env python
import sys
from stock_portfolio_intelligence_system.crew import StockPortfolioIntelligenceSystemCrew

# This main file is intended to be a way for your to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'stock_symbols': 'sample_value',
        'investment_strategy': 'sample_value',
        'telegram_chat_id': 'sample_value',
        'telegram_bot_token': 'sample_value'
    }
    StockPortfolioIntelligenceSystemCrew().crew().kickoff(inputs=inputs)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'stock_symbols': 'sample_value',
        'investment_strategy': 'sample_value',
        'telegram_chat_id': 'sample_value',
        'telegram_bot_token': 'sample_value'
    }
    try:
        StockPortfolioIntelligenceSystemCrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        StockPortfolioIntelligenceSystemCrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        'stock_symbols': 'sample_value',
        'investment_strategy': 'sample_value',
        'telegram_chat_id': 'sample_value',
        'telegram_bot_token': 'sample_value'
    }
    try:
        StockPortfolioIntelligenceSystemCrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: main.py <command> [<args>]")
        sys.exit(1)

    command = sys.argv[1]
    if command == "run":
        run()
    elif command == "train":
        train()
    elif command == "replay":
        replay()
    elif command == "test":
        test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
