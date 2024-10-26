import os
import subprocess
import logging
import json
import tempfile
import pyttsx3
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QTextEdit, QProgressBar, QApplication
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QTextCursor

from llms.gemini import GeminiService
import numpy as np
from llms.groq import GroqService
from config import GEMINI_API_KEY, GROQ_API_KEY
from voice_recognition_thread import VoiceRecognitionThread
from rag_service import RAGService
from create_embeddings import EmbeddingProvider
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TTSThread(QThread):
    finished = pyqtSignal()  # Add finished signal

    def __init__(self, text):
        super().__init__()
        self.text = text

    def run(self):
        try:
            engine = pyttsx3.init()
            engine.say(self.text)
            engine.runAndWait()
            engine.stop()
        finally:
            self.finished.emit()  # Emit finished signal when done

class VoiceAssistant(QMainWindow):
    def __init__(self, llm_service='gemini', embeddings_dir='embeddings'):
        super().__init__()
        
        # Initialize status states
        self.status_states = {
            'IDLE': "Idle",
            'LISTENING': "Listening for voice input...",
            'PROCESSING': "Processing command...",
            'GENERATING': "Generating script...",
            'VALIDATING': "Validating script...",
            'EXECUTING': "Executing command...",
            'INTERPRETING': "Interpreting results...",
            'SPEAKING': "Speaking response...",
            'ERROR': "Error occurred",
            'INITIALIZING': "Initializing microphone..."
        }
        self.current_status = 'IDLE'

        # Set up the main window
        self.setWindowTitle("Smart Voice Assistant")
        self.setGeometry(100, 100, 800, 600)
        self.setup_ui()
        
        # Initialize voice recognition thread
        self.voice_thread = VoiceRecognitionThread()
        self.voice_thread.status_update.connect(self.update_status)
        self.voice_thread.command_received.connect(self.process_command)
        self.voice_thread.listening_stopped.connect(self.on_listening_stopped)

        # Initialize LLM service
        if llm_service == 'gemini':
            self.llm = GeminiService(GEMINI_API_KEY)
        elif llm_service == 'groq':
            self.llm = GroqService(GROQ_API_KEY)
        else:
            raise ValueError(f"Unsupported LLM service: {llm_service}")

        self.llm.initialize()

        # Initialize RAG service
        try:
            self.rag = RAGService(embeddings_dir)
            self.rag.load_index()
        except FileNotFoundError:
            logging.error(
                "Embeddings not found. Please run create_embeddings.py first with your context file."
            )
            raise

        self.embedding_provider = EmbeddingProvider(llm_service)

        # Set up progress bar timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)

        # Initialize allowed commands
        self.initialize_allowed_commands()

        # Initialize temporary speech file
        self.speech_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
        
        logging.info("Voice Assistant initialized")

    def setup_ui(self):
        """Set up the user interface"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2C3E50;
            }
            QLabel {
                color: #ECF0F1;
                font-size: 16px;
            }
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 10px;
                font-size: 16px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QProgressBar {
                border: 2px solid #3498DB;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498DB;
            }
        """)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create header layout
        header_layout = QHBoxLayout()
        
        # Create and set up status label
        self.status_label = QLabel(self.status_states['IDLE'])
        self.status_label.setAlignment(Qt.AlignCenter)
        header_layout.addWidget(self.status_label)

        # Create and set up listen button
        self.listen_button = QPushButton("Start Listening")
        self.listen_button.clicked.connect(self.start_listening)
        header_layout.addWidget(self.listen_button)

        main_layout.addLayout(header_layout)

        # Create and set up progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Create and set up terminal
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setStyleSheet("""
            QTextEdit {
                background-color: #000000;
                color: #FFFFFF;
                border: none;
                font-family: 'DejaVu Sans Mono', monospace;
                font-size: 16px;
                padding: 10px;
            }
        """)
        main_layout.addWidget(self.terminal)

        # Initialize terminal prompt
        self.terminal.append("$ ")
        self.terminal.moveCursor(QTextCursor.End)

    def initialize_allowed_commands(self):
        """Initialize the set of allowed commands"""
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
            'alias', 'unalias', 'factor', 'yes', 'shutdown', 'reboot', 'kill', 'killall',
            'history', 'clear', 'exit', 'logout', 'su', 'sudo', 'passwd', 'useradd',
        ])

    def start_listening(self):
        """Start the voice recognition process"""
        self.progress_bar.setValue(0)
        self.timer.start(40)
        self.voice_thread.start()
        self.update_status(self.status_states['INITIALIZING'])
        self.listen_button.setEnabled(False)
        self.listen_button.setText("Listening...")

    def on_listening_stopped(self):
        """Handle cleanup when listening stops"""
        self.timer.stop()
        self.progress_bar.setValue(0)
        self.listen_button.setEnabled(True)
        self.listen_button.setText("Start Listening")
        if self.current_status == self.status_states['LISTENING']:
            self.update_status(self.status_states['IDLE'])

    def update_progress(self):
        """Update the progress bar"""
        value = self.progress_bar.value() + 1
        if value > 100:
            self.timer.stop()
            self.progress_bar.setValue(0)
            self.listen_button.setEnabled(True)
            self.listen_button.setText("Start Listening")
        else:
            self.progress_bar.setValue(value)

    def update_status(self, status):
        """Update status in UI and terminal"""
        self.current_status = status
        self.status_label.setText(status)
        self.terminal_print(f"Status: {status}")
        QApplication.processEvents()  # Force UI update

    def terminal_print(self, text):
        """Print text to the terminal widget"""
        self.terminal.moveCursor(QTextCursor.End)
        self.terminal.insertPlainText(text + "\n")
        self.terminal.moveCursor(QTextCursor.End)
        self.terminal.ensureCursorVisible()
        self.terminal.append("$ ")
        self.terminal.moveCursor(QTextCursor.End)

    def process_command(self, command):
        """Process the received voice command"""
        logging.info(f"Processing command: {command}")
        self.update_status(self.status_states['PROCESSING'])
        self.terminal_print(f"Received command: {command}")
        
        try:
            # Get context from RAG model
            self.update_status(self.status_states['GENERATING'])
            context = self.rag.get_relevant_context(command)
            
            # Generate bash script
            bash_script, description = self.llm.generate_bash_script(command, context)
            logging.debug(f"Generated bash script: {bash_script}")
            logging.debug(f"Description: {description}")
            
            self.terminal_print(f"Description: {description}")
            
            # Validate script
            self.update_status(self.status_states['VALIDATING'])
            self.terminal_print("Validating script...")
            
            if self.validate_script(bash_script):
                # Execute in sandbox
                self.update_status(self.status_states['EXECUTING'])
                self.terminal_print("Executing command in sandbox...")
                command_output = self.execute_in_sandbox(bash_script)
                self.terminal_print(f"Raw output: {command_output}")
                
                # Interpret output
                self.update_status(self.status_states['INTERPRETING'])
                interpreted_response = self.llm.interpret_output(command, command_output)
                self.terminal_print(f"Interpreted response: {interpreted_response}")
                
                # Speak response
                self.update_status(self.status_states['SPEAKING'])
                self.speak_text(interpreted_response)
                
            else:
                self.update_status(self.status_states['ERROR'])
                error_message = "Script execution aborted due to security concerns."
                self.terminal_print(error_message)
                self.speak_text(error_message)
                
        except Exception as e:
            self.update_status(self.status_states['ERROR'])
            error_message = f"Error executing command: {str(e)}"
            self.terminal_print(error_message)
            self.speak_text(error_message)
            logging.error(error_message)

    def validate_script(self, script_content):
        """Validate the generated bash script"""
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
            result = subprocess.run(['shellcheck', '-s', 'bash', temp_file_path], 
                                 capture_output=True, text=True)
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
        """Execute the bash script in a sandboxed environment"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as temp_file:
            temp_file.write(script_content)
            temp_file_path = temp_file.name

        try:
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

    def speak_text(self, text):
        """Speak the given text using text-to-speech"""
        self.tts_thread = TTSThread(text)
        self.tts_thread.finished.connect(lambda: self.update_status(self.status_states['IDLE']))
        self.tts_thread.start()

    def closeEvent(self, event):
        """Handle application closing"""
        if hasattr(self, 'tts_thread') and self.tts_thread.isRunning():
            self.tts_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = VoiceAssistant()
    window.show()
    app.exec_()