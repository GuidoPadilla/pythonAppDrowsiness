from kivy.app import App
from kivy.uix.label import Label
import threading
import subprocess
import os
import sys
import time


class MyApp(App):
    def build(self):
        # Start the Flask server in a separate process
        threading.Thread(target=self.start_flask_server).start()
        return Label(text='Flask server is running')

    def start_flask_server(self):
        # Change to the directory where your Flask app is located

        # Run the Flask app as a separate process
        # Replace with your Flask app filename
        flask_process = subprocess.Popen([sys.executable, 'server_face.py'])

        # Wait for the Flask process to finish
        flask_process.wait()


if __name__ == '__main__':
    MyApp().run()
