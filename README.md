# Intelligent Image Processing System

An intelligent image processing system that combines deep learning and traditional image processing techniques. The system can process images based on natural language instructions and provides various image enhancement features.

## Features

- Natural Language Processing for Image Enhancement
- Auto White Balance
- Auto Exposure
- Noise Reduction
- Edge Detection
- Feature Detection
- Image Enhancement
- Image Segmentation

## Technology Stack

### Core Technologies
- Python 3.8+
- OpenCV
- NumPy
- scikit-image
- PyTorch
- Gradio (UI Framework)
- Google Gemini API (Natural Language Processing)

### Image Processing Technologies

#### 1. Auto White Balance
- **Implementation**: OpenCV
- **Methods**:
  - Gray World Algorithm
  - Perfect Reflector Method
- **Resources**:
  - [OpenCV Implementation](https://github.com/opencv/opencv/blob/master/modules/imgproc/src/color.cpp)
  - [Gray World Algorithm Paper](https://www.researchgate.net/publication/220494581_Color_Constancy_Using_Gray_World_Assumption)

#### 2. Auto Exposure
- **Implementation**: OpenCV + NumPy
- **Methods**:
  - Target Brightness Tracking
  - Dynamic Range Adjustment
- **Resources**:
  - [OpenCV Exposure Control](https://github.com/opencv/opencv/blob/master/modules/imgproc/src/color.cpp)
  - [Auto Exposure Algorithm Paper](https://www.researchgate.net/publication/221653473_Automatic_Exposure_Control_in_Low_Light_Environments)

#### 3. Image Enhancement
- **Implementation**: OpenCV + scikit-image
- **Methods**:
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Histogram Equalization
- **Resources**:
  - [OpenCV CLAHE Implementation](https://github.com/opencv/opencv/blob/master/modules/imgproc/src/clahe.cpp)
  - [CLAHE Algorithm Paper](https://www.researchgate.net/publication/220494581_Contrast_Limited_Adaptive_Histogram_Equalization)

#### 4. Noise Reduction
- **Implementation**: OpenCV
- **Methods**:
  - Non-Local Means Denoising
  - Bilateral Filter
  - Gaussian Filter
- **Resources**:
  - [OpenCV Non-Local Means](https://github.com/opencv/opencv/blob/master/modules/photo/src/nlmeans.cpp)
  - [Non-Local Means Paper](https://www.researchgate.net/publication/220494581_A_Non-Local_Algorithm_for_Image_Denoising)

#### 5. Edge Detection
- **Implementation**: OpenCV + scikit-image
- **Methods**:
  - Canny
  - Sobel
  - Prewitt
- **Resources**:
  - [OpenCV Canny Implementation](https://github.com/opencv/opencv/blob/master/modules/imgproc/src/canny.cpp)
  - [Canny Edge Detection Paper](https://www.researchgate.net/publication/220494581_A_Computational_Approach_to_Edge_Detection)

#### 6. Feature Detection
- **Implementation**: OpenCV
- **Methods**:
  - SIFT
  - ORB
  - FAST
- **Resources**:
  - [OpenCV SIFT Implementation](https://github.com/opencv/opencv/blob/master/modules/features2d/src/sift.cpp)
  - [SIFT Algorithm Paper](https://www.researchgate.net/publication/220494581_Distinctive_Image_Features_from_Scale-Invariant_Keypoints)

## System Requirements

- Python 3.8 or higher
- Google Gemini API Key
- CUDA-capable GPU (optional, for faster processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/intelligent-image-processing.git
cd intelligent-image-processing
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

1. Start the application:
```bash
python src/main.py
```

2. Open the displayed URL in your browser (usually http://localhost:7860)

3. Choose processing method:
   - **Natural Language Instructions**:
     - Check "Use Natural Language Instructions"
     - Enter instructions in Chinese, e.g.:
       - "Remove noise and auto white balance"
       - "Make the image brighter with higher contrast"
       - "Make the image sharper and increase saturation"
   
   - **Manual Processing Options**:
     - Uncheck "Use Natural Language Instructions"
     - Select processing options:
       - "Auto Exposure"
       - "Auto White Balance"
       - "Edge Detection" (Methods: Canny, Sobel, Prewitt)
       - "Feature Detection" (Methods: SIFT, ORB, FAST)
       - "Enhancement" (Methods: CLAHE, Histogram Equalization)
       - "Noise Reduction" (Methods: NLM, Bilateral, Gaussian)
       - "Image Segmentation" (Methods: Watershed, K-means)

## Project Structure

```
.
├── src/
│   ├── image_processing.py
│   ├── auto_exposure.py
│   ├── auto_focus.py
│   ├── auto_white_balance.py
│   ├── vision_algorithms.py
│   ├── isp_pipeline.py
│   ├── ui/
│   │   ├── image_processing_interface.py
│   │   └── cli_interface.py
│   ├── llm_interface/
│   │   └── gemini_interface.py
│   └── main.py
├── requirements.txt
├── .env.example
├── setup.py
└── README.md
```

## Additional Resources

### Learning Resources
- [OpenCV Documentation](https://docs.opencv.org/4.x/)
- [scikit-image Documentation](https://scikit-image.org/docs/stable/)
- [Python Image Processing Tutorial](https://opencv-python-tutroals.readthedocs.io/en/latest/)

### Related Papers
- [Image Processing Algorithms Review](https://www.researchgate.net/publication/220494581_Image_Processing_Algorithms_and_Applications)
- [Computer Vision Algorithms](https://www.researchgate.net/publication/220494581_Computer_Vision_Algorithms_and_Applications)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

- Name: Your Name
- Student ID: Your ID
- Department: Your Department

## Acknowledgments

- OpenCV community
- scikit-image team
- Google Gemini API team 