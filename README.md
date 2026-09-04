# SMS Spam Detection 📱

## About the Project

This project is a Machine Learning-based **SMS Spam Detection** system. It classifies SMS messages into two categories:

* **Ham** – Normal/legitimate messages
* **Spam** – Unwanted or promotional messages

The model learns from a dataset of SMS messages and predicts whether a new message is spam or not.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

## Dataset

The project uses the `spam.csv` dataset, which contains SMS messages labeled as **spam** or **ham**.

## Machine Learning

The SMS messages are converted into numerical features using **TF-IDF Vectorization**. A Machine Learning classification algorithm is then trained to identify spam messages.

## How to Run

1. Clone this repository:

```bash
git clone  https://github.com/ananya3498-cell/SMS-Spam-Detection.git
```

2. Open the project folder in VS Code.

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Run the Python program:

```bash
python spam_detector.py
```

## Result

The trained model can predict whether an SMS message is **Spam** or **Ham** based on the text of the message.

## Future Improvements

* Create a simple web interface for users to enter messages
* Improve model accuracy
* Compare different Machine Learning algorithms
* Deploy the project as a web application

## Author

Ananya Priyadarshini
