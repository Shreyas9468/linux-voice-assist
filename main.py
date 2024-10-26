import sys
from PyQt5.QtWidgets import QApplication
from voice_assistant import VoiceAssistant

if __name__ == "__main__":
    app = QApplication(sys.argv)
    assistant = VoiceAssistant(llm_service="groq", embeddings_dir='/home/sidd/dev/lva/embeddings')
    assistant.show()
    sys.exit(app.exec_())
