================================================
  AI BATTLE ARENA - README
================================================

Two AIs fight in a 10x10 grid arena!
They can attack, collect items, and discover SECRET moves.

================================================
QUICK START
================================================

WINDOWS:
1. Double-click "setup.bat"
2. Run: python train_battle.py
3. Watch: python watch_battle.py

LINUX/MAC:
1. Run: bash setup.sh
2. Run: python3 train_battle.py
3. Watch: python3 watch_battle.py

================================================
WHAT'S INCLUDED
================================================

setup.bat / setup.sh
  - Installs Python dependencies automatically

train_battle.py
  - Trains 2 AIs for 10,000 episodes
  - Shows action usage statistics every 500 episodes
  - Saves trained models as .pth files

watch_battle.py
  - Loads trained models
  - Shows 5 live battles in terminal
  - Real-time visualization with colors

================================================
GAME MECHANICS
================================================

ARENA: 10x10 grid
HEALTH: 100 HP each
WIN: Reduce opponent to 0 HP or have more HP at timeout

STANDARD ACTIONS:
  [0-3] Move (Up, Down, Left, Right)
  [4]   Attack - Range 2, Damage 15, Cooldown 3
  [5]   Use Item

ITEMS (spawn on ground):
  H (Green)  = Health Pack (+30 HP)
  A (Red)    = Attack Boost
  S (Yellow) = Speed Boost
  D (Cyan)   = Shield

SECRET ACTIONS (AI must discover!):
  [6] Dodge    - Quick escape, +5 reward, CD 5
  [7] Teleport - Random warp, +8 reward, CD 10, 60% success
  [8] Counter  - Reflect damage, +4 reward, CD 7
  [9] Berserk  - 2x damage, +6 reward, CD 15

================================================
SCALABILITY (Future)
================================================

This is a PROOF OF CONCEPT for:
- Players training their own AI models
- Uploading .pth files to compete
- Tournament brackets
- Different arena maps
- ELO rankings
- Spectator mode

The .pth files are PORTABLE - you can share them!

================================================
TRAINING TIPS
================================================

- Training takes ~30-60 minutes
- Watch the "Secret Actions" usage increase over time
- AIs start random, get smarter over episodes
- Epsilon (ε) shows exploration rate
- Action bars show what AIs learned

================================================
TROUBLESHOOTING
================================================

"Module not found":
  - Run setup.bat or setup.sh first

"File not found" when watching:
  - Train first with train_battle.py

Slow performance:
  - Reduce episodes in train_battle.py
  - Or run on faster hardware

================================================
CUSTOMIZATION
================================================

Easy changes in train_battle.py:
- Line 12: Arena size (default 10)
- Line 397: Episodes (default 10000)
- Line 398: Batch size (default 64)
- Line 156-160: Add new secret actions!
- Line 28-32: Modify rewards

================================================

Have fun! Watch your AIs discover the secrets! 🎮⚔️

Created for competitive AI training experiments.
Next step: Online tournaments!

================================================
