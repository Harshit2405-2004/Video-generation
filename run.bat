@echo off
echo Activating virtual environment...
call venv\Scripts\activate

echo Running AI Video Generator...
python main.py
pause
