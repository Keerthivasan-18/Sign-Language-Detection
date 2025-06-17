# 🧠 Sign Language Detection using MediaPipe & CNN

This project implements **real-time detection of American Sign Language (ASL)** using computer vision and machine learning techniques. The model is trained to recognize ASL alphabets (A–Z) from webcam or image input using **MediaPipe** for hand tracking and **Convolutional Neural Networks (CNN)** for classification.

---

## 📁 Project Structure

```
├── dataset/                   # ASL alphabet images (A-Z folders)
├── processed_data/            # Processed images after hand detection
├── train.py                   # CNN training and evaluation
├── webCam.py                 # Inference using webcam
├── asl_model.h5               # Trained CNN model
├── README.md                  # Project documentation
```

---

## 🔧 Features

- ✋ Real-time hand detection using **MediaPipe**
- 🎨 Background removal and image normalization
- 🧠 CNN-based classifier for ASL alphabets
- 🎥 Webcam-based prediction in real time
- 📸 Support for individual image testing

---

## 🧪 Technologies Used

- **Python 3.10**
- **TensorFlow / Keras**
- **OpenCV**
- **MediaPipe**
- **NumPy**
- **imgaug**
- **Matplotlib**
- **Jupyter Notebook**

---

## 📊 Dataset

Dataset: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)  
- 26 folders (`A-Z`)  
- 3000+ images per letter  
- Each image contains a single hand pose for an ASL letter

---

## 🚀 How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Preprocess dataset with MediaPipe**
   ```bash
   python mediapipe_hand_preprocess.py
   ```

3. **Train the CNN model**
   ```bash
   python model.py
   ```

4. **Run real-time prediction**
   ```bash
   python predict.py
   ```

---

## 📸 Sample Output

| Sign | Prediction |
|------|------------|
| ✊   | A          |
| 🖐️  | B          |
| 🤟  | L          |

---

## 💡 Future Improvements

- 🔤 Sentence-level detection (RNN/Transformer)
- 🧾 Live subtitle generator from signs
- 🌐 Web interface with Streamlit or Flask
- 📱 Deploy on mobile using TensorFlow Lite

---

## 👨‍💻 Author

**Keerthivasan G**

---

## 📬 Contact

For any inquiries or contributions, feel free to reach out:  
📧 **keerthivasang50@gmail.com**
