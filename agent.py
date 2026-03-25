"""
TODO: build agent loop

1. take starting url and goal/instruction
2. call get_observation() to get initial observation
3. build system prompt with observation, goal, and history
4. call llm to get action (need to impl openai calls in llm.py)
5. parse and validate action
6. execute action
7. log everything and repeat until stop or max_steps
"""