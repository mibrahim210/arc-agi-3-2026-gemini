import importlib.util, sys
from pathlib import Path
import arc_agi as arc

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

import arc_agi
from arc_agi import OperationMode

spec = importlib.util.spec_from_file_location('my_agent', ROOT / 'agent' / 'my_agent.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

from arc_agi import OperationMode

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)

def diagnose_game(game_id, max_steps=30):
    print(f"\n==================== DIAGNOSING {game_id} ====================")
    env = arc.make(game_id)
    mod.MyAgent.MAX_ACTIONS = max_steps
    agent = mod.MyAgent(
        card_id="local-dev",
        game_id=game_id,
        agent_name=f"MyAgent.local.{game_id}",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=env,
        tags=["local-dev"],
    )
    
    orig_choose = agent.choose_action
    step_history = []
    
    def hooked_choose(frames, latest_frame):
        action = orig_choose(frames, latest_frame)
        r = getattr(action, 'reasoning', {}) or {}
        prev = r.get('previous_event', {}) or {}
        step_history.append({
            'step': len(step_history),
            'action': action.name,
            'data': getattr(action, 'data', None),
            'source': r.get('source'),
            'stage': r.get('stage'),
            'predicted': r.get('predicted'),
            'entity_conf': r.get('entity_confidence'),
            'field_mode': r.get('field_mode'),
            'noop_streak': agent.memory.no_op_streak,
            'noop': prev.get('noop'),
            'changed': prev.get('changed'),
            'death': prev.get('death'),
            'progress': prev.get('progress'),
        })
        return action

    agent.choose_action = hooked_choose
    try:
        agent.main()
    except Exception as e:
        print(f"Exception: {e}")
        
    print(f"Total Steps Recorded: {len(step_history)}")
    for item in step_history[:25]:
        print(f"Step {item['step']:02d}: Act={item['action']:7s} Src={str(item['source']):25s} Stg={str(item['stage']):25s} NoopStrk={item['noop_streak']} PrevChg={item['changed']} PrevNoop={item['noop']} PrevDth={item['death']}")
    if len(step_history) > 25:
        print("...")
        for item in step_history[-5:]:
            print(f"Step {item['step']:02d}: Act={item['action']:7s} Src={str(item['source']):25s} Stg={str(item['stage']):25s} NoopStrk={item['noop_streak']} PrevChg={item['changed']} PrevNoop={item['noop']} PrevDth={item['death']}")

for gid in ['wa30', 'r11l', 'ka59']:
    diagnose_game(gid, 25)
