# Mood Wave Flutter Project

## Introduction
Mood Wave is a mobile application designed to detect and track emotions using machine learning models integrated with Flutter. This project provides real-time emotion analysis through camera input, audio recording, and visualization of emotion history.

## Setup Instructions

### Prerequisites
- **Flutter SDK**: Make sure you have Flutter installed. If not, follow the [Flutter installation guide](https://flutter.dev/docs/get-started/install).
- **Development Environment**: Android Studio or Visual Studio Code with Flutter and Dart plugins installed.
- **Firebase Account**: You'll need a Firebase project for authentication and cloud services.

### Getting Started

1. **Clone the repository:**
   ```bash
Update IPC configuration:
while you are on Dashboard enter your ip in textfield 
just ip without http://
Replace the URL with PC's URL using ipcnofig.

Install dependencies:
Run the app:

Connect your device/emulator.
Execute the following command to run the app:
    ```bash
    flutter pub get
    flutter run

Live Emotion Detection: Real-time analysis of emotions using the device's camera.
Audio Recording: Record audio and analyze emotions.
Emotion History: View history of detected emotions.
Settings: Configure app preferences including emotion detection settings.
lib/: Contains all Dart code for the project.
screens/: Screen definitions.
services/: Backend services including authentication and emotion detection logic.
utils/: Reusable UI components.