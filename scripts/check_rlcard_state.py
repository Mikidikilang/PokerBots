from src.env.wrappers import RLCardWrapper, WrapperConfig
env = RLCardWrapper(config=WrapperConfig(num_players=2))
state = env.reset()
print("=== AFTER RESET ===")
print("State keys:", list(state.keys()))
print("hand:", state.get('hand'))
print("public_cards:", state.get('public_cards'))

# Now do a step
action = state['legal_actions'][0]
print(f"\n=== TAKING ACTION {action} ===")
step_result = env.step(action)
print("TYPE OF STEP RESULT:", type(step_result))
print("LENGTH OF STEP RESULT:", len(step_result) if isinstance(step_result, (tuple, list)) else "N/A")
if isinstance(step_result, (tuple, list)):
    print("Element types:", [type(x).__name__ for x in step_result])
    state_step = step_result[0]
    print("\nStep state type:", type(state_step))
    if isinstance(state_step, dict):
        print("Step state keys:", list(state_step.keys()))
        print("hand:", state_step.get('hand'))
        print("public_cards:", state_step.get('public_cards'))
