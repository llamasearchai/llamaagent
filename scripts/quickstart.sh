#!/bin/bash
# LlamaAgent Master Program Quick Start Script

echo "LAUNCH: LlamaAgent Master Program Quick Start"
echo "========================================"

# Check Python version
echo "Checking Python version..."
python3 --version

# Set up environment
echo -e "\nSetting up environment..."
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo -e "\nInstalling dependencies..."
pip install -e . 2>/dev/null || echo "Note: Some dependencies may need to be installed manually"

# Check OpenAI key
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "\nWARNING:  Warning: OPENAI_API_KEY not set. OpenAI integration will be disabled."
    echo "To enable OpenAI integration, run:"
    echo "  export OPENAI_API_KEY='your-api-key'"
fi

# Display available commands
echo -e "\nLIST: Available Commands:"
echo "====================="
echo ""
echo "1. Start API Server:"
echo "   python3 llamaagent_master_program.py server"
echo ""
echo "2. Execute a Task:"
echo "   python3 llamaagent_master_program.py execute \"Your task description\""
echo ""
echo "3. Run Demo:"
echo "   python3 llamaagent_master_program.py demo"
echo ""
echo "4. Monitor System:"
echo "   python3 llamaagent_master_program.py monitor"
echo ""
echo "5. Run Tests:"
echo "   python3 test_master_program.py"
echo ""
echo "6. View Help:"
echo "   python3 llamaagent_master_program.py --help"
echo ""

# Ask user what to do
echo "What would you like to do?"
echo "1) Start API Server"
echo "2) Run Demo"
echo "3) Run Tests"
echo "4) Exit"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo -e "\nStarting API Server on http://localhost:8000..."
        echo "Press Ctrl+C to stop"
        python3 llamaagent_master_program.py server
        ;;
    2)
        echo -e "\nRunning Demo..."
        python3 llamaagent_master_program.py demo
        ;;
    3)
        echo -e "\nRunning Tests..."
        python3 test_master_program.py
        ;;
    4)
        echo "Exiting..."
        ;;
    *)
        echo "Invalid choice. Exiting..."
        ;;
esac
