import matplotlib
matplotlib.use('Agg')  # Needed for running matplotlib without a display server
import io
import base64
import pandas as pd
import preprocessor
import helper
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length
from flask_migrate import Migrate
from textblob import TextBlob  # for sentiment analysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation  # for topic modeling

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
migrate = Migrate(app, db)

# Define User and Summary models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    summaries = db.relationship('Summary', backref='author', lazy=True)

class Summary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Define Flask Forms
class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField("Register")

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField("Login")

# Function to convert plot to base64 encoding
def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user is None:
            hashed_password = generate_password_hash(form.password.data, method='sha256')
            new_user = User(username=form.username.data, password=hashed_password)
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Username already exists. Please choose a different one.', 'danger')
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('index'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html', form=form)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def index():
    if request.method == 'POST':
        file = request.files['chat_file']
        if file:
            file_path = file.filename
            file.save(file_path)
            return redirect(url_for('analyze', filename=file_path))
    return render_template('index.html', username=current_user.username)

@app.route('/summaries')
@login_required
def summaries():
    user_summaries = Summary.query.filter_by(user_id=current_user.id).all()
    return render_template('summaries.html', summaries=user_summaries)


@app.route('/summary/<int:summary_id>')
@login_required
def summary(summary_id):
    summary = Summary.query.get_or_404(summary_id)
    summary_content = eval(summary.content)  # Convert the string back to a dictionary
    return render_template('summary.html', **summary_content)

@app.route('/summary/delete/<int:summary_id>', methods=['POST'])
@login_required
def delete_summary(summary_id):
    summary = Summary.query.get_or_404(summary_id)
    if summary.user_id != current_user.id:
        flash('You do not have permission to delete this summary.', 'danger')
        return redirect(url_for('summaries'))
    
    db.session.delete(summary)
    db.session.commit()
    flash('Summary deleted successfully.', 'success')
    return redirect(url_for('summaries'))

@app.route('/analyze/<filename>', methods=['GET'])
@login_required
def analyze(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = file.read()

    df = preprocessor.preprocess(data)
    selected_user = 'Overall'

    num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

    # Sentiment Analysis
    sentiments = helper.sentiment_analysis(selected_user, df)
    sentiment_plot = plot_sentiment(sentiments)

    # Topic Modeling
    topics = helper.topic_modeling(selected_user, df)
    topic_plots = []
    for topic in topics:
        topic_plot = plot_topic(topic)
        topic_plots.append(topic_plot)

    # Monthly Timeline
    timeline = helper.monthly_timeline(selected_user, df)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(timeline['time'], timeline['message'], color='green')
    plt.xticks(rotation=45)
    monthly_timeline_plot = plot_to_base64(fig)
    plt.close(fig)

    # Daily Timeline
    daily_timeline = helper.daily_timeline(selected_user, df)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
    plt.xticks(rotation=45)
    daily_timeline_plot = plot_to_base64(fig)
    plt.close(fig)

    # Most Busy Day
    busy_day = helper.week_activity_map(selected_user, df)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(busy_day.index, busy_day.values, color='purple')
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Number of Messages')
    busy_day_plot = plot_to_base64(fig)
    plt.close(fig)

    # Most Busy Month
    busy_month = helper.month_activity_map(selected_user, df)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(busy_month.index, busy_month.values, color='orange')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Messages')
    busy_month_plot = plot_to_base64(fig)
    plt.close(fig)

    # Activity Heatmap
    user_heatmap = helper.activity_heatmap(selected_user, df)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(user_heatmap, ax=ax)
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Day of Week')
    heatmap_plot = plot_to_base64(fig)
    plt.close(fig)
    
     # Wordcloud
    df_wc = helper.create_wordcloud(selected_user, df)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(df_wc, interpolation='bilinear')
    ax.axis('off')
    wordcloud_plot = plot_to_base64(fig)
    plt.close(fig)
    
    # Most Common Words
    most_common_df = helper.most_common_words(selected_user, df)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.barh(most_common_df[0], most_common_df[1], color='blue')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Words')
    common_words_plot = plot_to_base64(fig)
    plt.close(fig)

    # Emoji Analysis
    emoji_df = helper.emoji_helper(selected_user, df)
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct='%1.1f%%', startangle=140)
    emoji_pie_plot = plot_to_base64(fig)
    plt.close(fig)

    summary_content = {
        'selected_user': selected_user,
        'num_messages': num_messages,
        'words': words,
        'num_media_messages': num_media_messages,
        'num_links': num_links,
        'sentiment_plot': sentiment_plot,
        'topic_plots': topic_plots,
        'monthly_timeline_plot': monthly_timeline_plot,
        'daily_timeline_plot': daily_timeline_plot,
        'busy_day_plot': busy_day_plot,
        'busy_month_plot': busy_month_plot,
        'wordcloud_plot': wordcloud_plot,
        'heatmap_plot': heatmap_plot,
        'common_words_plot': common_words_plot,
        'emoji_df': emoji_df.to_dict(),
        'emoji_pie_plot': emoji_pie_plot
    }

    new_summary = Summary(content=str(summary_content), user_id=current_user.id)
    db.session.add(new_summary)
    db.session.commit()

    return render_template('result.html', **summary_content)

def plot_sentiment(sentiments):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.hist(sentiments, bins=30, alpha=0.7, color='skyblue')
    ax.set_xlabel('Sentiment Polarity')
    ax.set_ylabel('Frequency')
    sentiment_plot = plot_to_base64(fig)
    plt.close(fig)
    return sentiment_plot

def plot_topic(topic):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(range(len(topic)), topic, color='steelblue')
    ax.set_xlabel('Top Words')
    ax.set_ylabel('Frequency')
    topic_plot = plot_to_base64(fig)
    plt.close(fig)
    return topic_plot

if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)
