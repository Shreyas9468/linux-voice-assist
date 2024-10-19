import sys
from PyQt5.QtWidgets import QApplication
from voice_assistant import VoiceAssistant

if __name__ == "__main__":
    app = QApplication(sys.argv)
    assistant = VoiceAssistant()
    assistant.show()
    sys.exit(app.exec_())
