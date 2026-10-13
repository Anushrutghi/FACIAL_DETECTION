from flask import Flask, request, render_template
import cv2
import numpy as np
from detection.face_detect import detect_faces

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            try:
                in_memory_file = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

                if img is None:
                    return render_template('index.html', img_data=None, error='Invalid image file.')

                faces = detect_faces(img)

                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                _, img_encoded = cv2.imencode('.jpg', img)
                img_data = img_encoded.tobytes()

                return render_template(
                    'index.html',
                    img_data=img_data,
                    error=None,
                    face_count=len(faces)
                )
            except Exception as e:
                return render_template('index.html', img_data=None, error=f'Error: {str(e)}')
    
    return render_template('index.html', img_data=None, error=None, face_count=0)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)