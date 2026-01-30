#!/bin/bash
echo "================================================"
echo " AI BATTLE ARENA - Setup"
echo "================================================"
echo ""
echo "Installing dependencies..."
echo ""

if ! command -v python3 &> /dev/null
then
    echo "ERROR: Python 3 is not installed!"
    echo "Please install Python 3.8+ first"
    exit 1
fi

pip3 install -r requirements.txt

echo ""
echo "================================================"
echo " Setup Complete!"
echo "================================================"
echo ""
echo "To train AIs:      python3 train_battle.py"
echo "To watch battle:   python3 watch_battle.py"
echo ""
