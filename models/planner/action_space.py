# models/planner/action_space.py

"""
Action space definition for ReAct-IR.

Planner / ToolBank / ReAct-IR core 모두가
이 파일을 기준으로 action token을 공유한다.
"""

# -------------------------
# Core actions
# -------------------------
A_DEDROP  = "A_DEDROP"    # raindrop / raindot removal
A_DEBLUR  = "A_DEBLUR"    # motion / defocus blur
A_DERAIN  = "A_DERAIN"    # rain streak
A_DEHAZE  = "A_DEHAZE"    # haze / fog
A_DESNOW  = "A_DESNOW"
# -------------------------
# Meta / control actions
# -------------------------
A_HYBRID  = "A_HYBRID"    # mixed degradation / safe exploration
A_STOP    = "A_STOP"      # terminate restoration
A_ABORT   = "A_ABORT"     # fail-safe / fallback


# -------------------------
# Optional: registry
# -------------------------
ALL_ACTIONS = [
    A_DEDROP,
    A_DEBLUR,
    A_DERAIN,
    A_DESNOW,
    A_DEHAZE,
    A_HYBRID,
    A_STOP,
    A_ABORT,
]


if __name__ == "__main__":
    print("[DEBUG] Action space:")
    for a in ALL_ACTIONS:
        print(" ", a)
    print("[DEBUG] action_space OK ✅")
