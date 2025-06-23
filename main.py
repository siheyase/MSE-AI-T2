from models.agent import get_agent

if __name__ == "__main__":

    agent = get_agent()
    user_input = "什么是病毒性心肌炎？"
    #user_input = "中国的首都在哪？"
    res = agent.run(user_input)
