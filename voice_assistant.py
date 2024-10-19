import os
import subprocess
import logging
import json
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QProgressBar
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QPalette, QTextCursor
import google.generativeai as genai
from voice_recognition_thread import VoiceRecognitionThread
from config import GEMINI_API_KEY, PROMPT_TEMPLATE

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

class VoiceAssistant(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Voice Assistant")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Status label
        self.status_label = QLabel("Idle")
        layout.addWidget(self.status_label)

        # Button to start listening
        self.listen_button = QPushButton("Start Listening")
        self.listen_button.clicked.connect(self.start_listening)
        layout.addWidget(self.listen_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Enhanced terminal
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setStyleSheet("""
            QTextEdit {
                background-color: #0C0C0C;
                color: #FFFFFF;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.terminal)

        # Set up the terminal prompt
        self.terminal.append("$ ")
        self.terminal.moveCursor(QTextCursor.End)

        # Voice recognition thread
        self.voice_thread = VoiceRecognitionThread()
        self.voice_thread.status_update.connect(self.update_status)
        self.voice_thread.command_received.connect(self.process_command)

        # Timer for progress bar
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)

        logging.info("Voice Assistant initialized")

    def start_listening(self):
        logging.info("Starting listening")
        self.progress_bar.setValue(0)
        self.timer.start(40)  # Update every 40ms for smooth progress (4000ms / 100)
        self.voice_thread.start()
        self.terminal_print("Listening for command...")

    def update_progress(self):
        value = self.progress_bar.value() + 1
        if value > 100:
            self.timer.stop()
            self.progress_bar.setValue(0)
        else:
            self.progress_bar.setValue(value)

    def update_status(self, status):
        logging.debug(f"Status updated: {status}")
        self.status_label.setText(status)
        self.terminal_print(f"Status: {status}")

    def terminal_print(self, text):
        self.terminal.moveCursor(QTextCursor.End)
        self.terminal.insertPlainText(text + "\n")
        self.terminal.moveCursor(QTextCursor.End)
        self.terminal.ensureCursorVisible()
        self.terminal.append("$ ")
        self.terminal.moveCursor(QTextCursor.End)

    def process_command(self, command):
        logging.info(f"Processing command: {command}")
        self.update_status("Processing")
        self.terminal_print(f"Received command: {command}")
        
        # Use Gemini API to convert command to bash script
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = PROMPT_TEMPLATE.format(query=command)
        response = model.generate_content(prompt)
        
        try:
            jsondata = json.loads(response.text)
            bash_script = jsondata.get('bash script')
            description = jsondata.get('description')
            
            logging.debug(f"Generated bash script: {bash_script}")
            logging.debug(f"Description: {description}")
            
            self.terminal_print(f"Description: {description}")
            self.terminal_print("Executing command...")
            
            # Write bash script to file
            with open('script.sh', 'w') as f:
                f.write(bash_script)
            
            # Make script executable and run it
            os.system('chmod +x script.sh')
            result = subprocess.run(['bash', 'script.sh'], capture_output=True, text=True)
            
            output = result.stdout if result.returncode == 0 else result.stderr
            self.terminal_print(f"Output: {output}")
            logging.info(f"Command executed. Output: {output}")
        except json.JSONDecodeError:
            error_message = "Error: Invalid JSON response from Gemini API"
            self.terminal_print(error_message)
            logging.error(error_message)
        except Exception as e:
            error_message = f"Error executing command: {str(e)}"
            self.terminal_print(error_message)
            logging.error(error_message)
        
        self.update_status("Idle")
