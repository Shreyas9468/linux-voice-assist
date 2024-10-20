import os
import subprocess
import logging
import json
import tempfile
import pyttsx3
# from gtts import gTTS
# from playsound import playsound
# from pydub import AudioSegment
# from pydub.generators import Sine
# import simpleaudio as sa
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QTextEdit, QProgressBar
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QPalette, QTextCursor
import google.generativeai as genai
from voice_recognition_thread import VoiceRecognitionThread
from config import GEMINI_API_KEY, PROMPT_TEMPLATE, OUTPUT_INTERPRETATION_PROMPT

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

class TTSThread(QThread):
    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        engine = pyttsx3.init()
        engine.say(self.text)
        engine.runAndWait()
        engine.stop()


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

        self.allowed_commands = set([
            'ls', 'echo', 'cat', 'grep', 'awk', 'sed', 'cut', 'sort', 'uniq',
            'wc', 'head', 'tail', 'find', 'date', 'pwd', 'whoami', 'uname', 'mkdir', 
            'rmdir', 'touch', 'cp', 'mv', 'less', 'nano', 'vim', 'more', 'diff', 
            'tar', 'gzip', 'gunzip', 'zip', 'unzip', 'ping', 'curl', 'wget', 'apt', 
            'dpkg', 'python', 'python3', 'g++', 'gcc', 'node', 'javac', 'java', 
            'ruby', 'df', 'du', 'free', 'top', 'htop', 'uptime', 'ps', 'id', 
            'hostname', 'cal', 'man', 'bc', 'time', 'xargs', 'tr', 'chmod', 'chown', 
            'tee', 'split', 'dmesg', 'iostat', 'vmstat', 'sar', 'lsof', 'who', 
            'last', 'mount', 'umount', 'blkid', 'fdisk', 'mkfs', 'ifconfig', 'netstat', 
            'ss', 'iptables', 'traceroute', 'nmap', 'which', 'locate', 'bc',
            'alias', 'unalias', 'factor', 'yes', 'shutdown', 'reboot', 'kill', 'killall'
        ])
        # self.tts_engine = pyttsx3.init()
        # for text-to-speech
        self.speech_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name


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

    def validate_script(self, script_content):
        # Step 1: Check against whitelist
        for line in script_content.split('\n'):
            command = line.strip().split()[0] if line.strip() else ''
            if command and command not in self.allowed_commands:
                self.terminal_print(f"Validation failed: '{command}' is not in the whitelist of allowed commands.")
                return False

        # Step 2: Use shellcheck for static analysis
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as temp_file:
            temp_file.write(script_content)
            temp_file_path = temp_file.name

        try:
            result = subprocess.run(['shellcheck', '-s', 'bash', temp_file_path], capture_output=True, text=True)
            if result.returncode != 0:
                self.terminal_print("ShellCheck found issues in the script:")
                self.terminal_print(result.stdout)
                return False
            else:
                self.terminal_print("ShellCheck validation passed.")
                return True
        finally:
            os.unlink(temp_file_path)

    def execute_in_sandbox(self, script_content):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as temp_file:
            temp_file.write(script_content)
            temp_file_path = temp_file.name

        try:
            # Use firejail to create a sandboxed environment
            result = subprocess.run([
                'firejail',
                '--noprofile',
                '--quiet',
                '--private',
                '--noroot',
                '--net=none',
                '--nosound',
                '--no3d',
                'bash',
                temp_file_path
            ], capture_output=True, text=True)

            return result.stdout if result.returncode == 0 else result.stderr
        finally:
            os.unlink(temp_file_path)

    def interpret_output(self, original_query, command_output):
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = OUTPUT_INTERPRETATION_PROMPT.format(
            query=original_query,
            output=command_output
        )
        response = model.generate_content(prompt)
        return response.text

    def speak_text(self, text):
        self.tts_thread = TTSThread(text)
        self.tts_thread.start()
        self.tts_thread.wait()  # Wait for the thread to finish



    def closeEvent(self, event):
        # Ensure all threads are stopped before closing
        if hasattr(self, 'tts_thread') and self.tts_thread.isRunning():
            self.tts_thread.wait()
        event.accept()


    def process_command(self, command):
        logging.info(f"Processing command: {command}")
        self.update_status("Processing")
        self.terminal_print(f"Received command: {command}")
        
        # First Gemini API call to convert command to bash script
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
            self.terminal_print("Validating script...")
            
            # Validate the script
            if self.validate_script(bash_script):
                self.terminal_print("Executing command in sandbox...")
                
                # Execute in sandbox
                command_output = self.execute_in_sandbox(bash_script)
                
                self.terminal_print(f"Raw output: {command_output}")
                
                # Second Gemini API call to interpret the output
                interpreted_response = self.interpret_output(command, command_output)
                
                self.terminal_print(f"Interpreted response: {interpreted_response}")
                logging.info(f"Command executed and interpreted. Response: {interpreted_response}")

                # Speak the interpreted response
                self.speak_text(interpreted_response)
            else:
                error_message = "Script execution aborted due to security concerns."
                self.terminal_print(error_message)
                self.speak_text(error_message)
        except json.JSONDecodeError:
            error_message = "Error: Invalid JSON response from Gemini API"
            self.terminal_print(error_message)
            self.speak_text(error_message)
            logging.error(error_message)
        except Exception as e:
            error_message = f"Error executing command: {str(e)}"
            self.terminal_print(error_message)
            self.speak_text(error_message)
            logging.error(error_message)
        
        self.update_status("Idle")

