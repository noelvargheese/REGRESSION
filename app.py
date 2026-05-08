import base64
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

sns.set_style('whitegrid')
app = Flask(__name__)
DATA_FILE = 'social_media_user_behavior.csv'


def load_social_media_data(path=DATA_FILE):
    df = pd.read_csv(path)
    df = df.dropna(
        subset=['gender', 'mood_while_scrolling', 'influencer_status', 'takes_social_media_breaks']
    )
    return df.reset_index(drop=True)


def build_label_encoders(df, categorical_cols):
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


def load_regression_model():
    df = load_social_media_data()
    categorical_cols = [
        'gender',
        'mood_while_scrolling',
        'influencer_status',
        'takes_social_media_breaks',
    ]
    raw_options = {
        col: sorted(df[col].astype(str).unique().tolist())
        for col in categorical_cols if col != 'takes_social_media_breaks'
    }

    encoded_df, encoders = build_label_encoders(df.copy(), categorical_cols)
    X = encoded_df[['mood_while_scrolling', 'influencer_status', 'gender']]
    y = encoded_df['takes_social_media_breaks']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'mse': float(mean_squared_error(y_test, y_pred)),
        'r2': float(r2_score(y_test, y_pred)),
        'sample_count': int(len(df)),
    }

    return {
        'raw_options': raw_options,
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'metrics': metrics,
        'data_sample': df.head(8),
    }


def decode_target_value(value, encoder):
    classes = list(encoder.classes_)
    index = int(round(value))
    index = max(0, min(index, len(classes) - 1))
    return classes[index]


def build_classification_report():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifiers = [
        ('Logistic Regression', LogisticRegression(max_iter=500)),
        ('KNN', KNeighborsClassifier(n_neighbors=5)),
        ('Naive Bayes', GaussianNB()),
        ('Decision Tree', DecisionTreeClassifier(random_state=42)),
        ('Support Vector Machine', SVC(kernel='rbf', probability=True)),
    ]

    results = []
    for name, clf in classifiers:
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        results.append(
            {
                'model': name,
                'accuracy': float(accuracy_score(y_test, y_pred)),
            }
        )

    return results


def render_plot_image(results):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [r['model'] for r in results]
    values = [r['accuracy'] for r in results]
    ax.bar(labels, values, color='#4c72b0')
    ax.set_ylim(0, 1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Iris Model Accuracy Comparison')
    ax.set_xticklabels(labels, rotation=30, ha='right')

    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format='png', dpi=100)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')


regression_data = load_regression_model()
classification_results = build_classification_report()
accuracy_plot = render_plot_image(classification_results)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    form_values = {
        'gender': regression_data['raw_options']['gender'],
        'mood_while_scrolling': regression_data['raw_options']['mood_while_scrolling'],
        'influencer_status': regression_data['raw_options']['influencer_status'],
    }

    if request.method == 'POST':
        mood = request.form.get('mood_while_scrolling', '')
        influencer = request.form.get('influencer_status', '')
        gender = request.form.get('gender', '')

        try:
            encoded_features = np.array(
                [
                    regression_data['encoders']['mood_while_scrolling'].transform([mood])[0],
                    regression_data['encoders']['influencer_status'].transform([influencer])[0],
                    regression_data['encoders']['gender'].transform([gender])[0],
                ]
            ).reshape(1, -1)
            scaled_features = regression_data['scaler'].transform(encoded_features)
            score = regression_data['model'].predict(scaled_features)[0]
            prediction = decode_target_value(score, regression_data['encoders']['takes_social_media_breaks'])
        except Exception as error:
            prediction = f'Error making prediction: {error}'

    return render_template(
        'index.html',
        prediction=prediction,
        form_values=form_values,
        regression_metrics=regression_data['metrics'],
        data_sample=regression_data['data_sample'].to_html(classes='table table-bordered table-sm', index=False, border=0),
        classification_results=classification_results,
        accuracy_plot=accuracy_plot,
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
