import matplotlib
matplotlib.use('Agg') 

from flask import Flask, render_template, request, redirect, url_for
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

def plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['chat_file']
        if file:
            file_path = file.filename
            file.save(file_path)
            return redirect(url_for('analyze', filename=file_path))
    return render_template('index.html')

@app.route('/analyze/<filename>', methods=['GET'])
def analyze(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        data = file.read()

    df = preprocessor.preprocess(data)

    selected_user = 'Overall'
    num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

    # Monthly Timeline
    timeline = helper.monthly_timeline(selected_user, df)
    fig, ax = plt.subplots()
    ax.plot(timeline['time'], timeline['message'], color='green')
    plt.xticks(rotation='vertical')
    monthly_timeline_plot = plot_to_base64(fig)
    plt.close(fig)

    # Daily Timeline
    daily_timeline = helper.daily_timeline(selected_user, df)
    fig, ax = plt.subplots()
    ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
    plt.xticks(rotation='vertical')
    daily_timeline_plot = plot_to_base64(fig)
    plt.close(fig)

    # Most busy day
    busy_day = helper.week_activity_map(selected_user, df)
    fig, ax = plt.subplots()
    ax.bar(busy_day.index, busy_day.values, color='purple')
    busy_day_plot = plot_to_base64(fig)
    plt.close(fig)

    # Most busy month
    busy_month = helper.month_activity_map(selected_user, df)
    fig, ax = plt.subplots()
    ax.bar(busy_month.index, busy_month.values, color='orange')
    busy_month_plot = plot_to_base64(fig)
    plt.close(fig)

    # Activity heatmap
    user_heatmap = helper.activity_heatmap(selected_user, df)
    fig, ax = plt.subplots()
    sns.heatmap(user_heatmap, ax=ax)
    heatmap_plot = plot_to_base64(fig)
    plt.close(fig)

    if selected_user == 'Overall':
        x, top_users = helper.most_busy_users(df)
        fig, ax = plt.subplots()
        ax.bar(x.index, x.values, color='red')
        busy_users_plot = plot_to_base64(fig)
        plt.close(fig)
    else:
        top_users = None
        busy_users_plot = None

    # Wordcloud
    df_wc = helper.create_wordcloud(selected_user, df)
    fig, ax = plt.subplots()
    ax.imshow(df_wc, interpolation='bilinear')
    ax.axis('off')
    wordcloud_plot = plot_to_base64(fig)
    plt.close(fig)

    # Most common words
    most_common_df = helper.most_common_words(selected_user, df)
    fig, ax = plt.subplots()
    ax.barh(most_common_df[0], most_common_df[1], color='blue')
    plt.xticks(rotation='vertical')
    common_words_plot = plot_to_base64(fig)
    plt.close(fig)

    # Emoji analysis
    emoji_df = helper.emoji_helper(selected_user, df)
    fig, ax = plt.subplots()
    ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct='%1.1f%%', startangle=140)
    emoji_pie_plot = plot_to_base64(fig)
    plt.close(fig)

    return render_template('result.html', selected_user=selected_user, num_messages=num_messages, words=words,
                           num_media_messages=num_media_messages, num_links=num_links,
                           monthly_timeline_plot=monthly_timeline_plot, daily_timeline_plot=daily_timeline_plot,
                           busy_day_plot=busy_day_plot, busy_month_plot=busy_month_plot, heatmap_plot=heatmap_plot,
                           busy_users_plot=busy_users_plot, top_users=top_users,
                           wordcloud_plot=wordcloud_plot, common_words_plot=common_words_plot,
                           emoji_df=emoji_df, emoji_pie_plot=emoji_pie_plot)

if __name__ == "__main__":
    app.run(debug=True)
