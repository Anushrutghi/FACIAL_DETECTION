# Facial Recognition Web Application

This project is a web application that utilizes face detection technology to identify faces in uploaded images. It is built using Flask and OpenCV.

## Project Structure

```
facial-recognition-app
├── app.py                # Main entry point of the web application
├── detection
│   └── face_detect.py    # Contains the face detection logic
├── templates
│   └── index.html        # HTML interface for image upload
├── static
│   └── style.css         # CSS styles for the web application
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd facial-recognition-app
   ```

2. **Create a virtual environment** (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Run the application**:
   ```
   python app.py
   ```

2. **Open your web browser** and go to `http://127.0.0.1:5000`.

3. **Upload an image** using the provided form. The application will process the image and display the detected faces.

## Dependencies

- Flask
- OpenCV

## License

This project is licensed under the MIT License. See the LICENSE file for more details.