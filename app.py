import streamlit as st
import json
from pathlib import Path
from datetime import datetime, timedelta
import plotly.express as px
import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
from fpdf import FPDF
from PIL import Image, ImageDraw
import spacy
from keybert import KeyBERT
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

from nlp_utils import get_emotion, get_summary

# Load models
nlp_ner = spacy.load('en_core_web_sm')
kw_model = KeyBERT(model='paraphrase-MiniLM-L6-v2')

# Path to the JSON file
DATA_FILE = Path("data/entries.json")
DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
if not DATA_FILE.exists():
    DATA_FILE.write_text("[]")

# Load existing entries
with open(DATA_FILE, "r") as f:
    entries = json.load(f)

st.set_page_config(layout="wide", page_title="Memory Lens")

# Daily prompt
st.markdown(
    f"💡 **Daily Mood Check-in** — how are you feeling today ({datetime.now().strftime('%A, %b %d')})? ✨"
)

# Sidebar Filters
st.sidebar.title("⚙️ Filters & Settings")
emotion_filter = st.sidebar.selectbox(
    "Filter by emotion",
    options=["All"] + sorted(list({e['emotion'] for e in entries}))
)
keyword = st.sidebar.text_input("🔍 Search by keyword:")

# Input new entry
st.title("🧠 Memory Lens — AI Journal Companion")
new_entry = st.text_area(
    "💬 Write about your day or a memory...",
    height=150
)
if st.button("✨ Save Entry"):
    if new_entry.strip():
        emotion = get_emotion(new_entry)
        summary = get_summary(new_entry)
        new_data = {
            "timestamp": datetime.now().isoformat(),
            "entry": new_entry,
            "emotion": emotion,
            "summary": summary
        }
        entries.append(new_data)
        with open(DATA_FILE, "w") as f:
            json.dump(entries, f, indent=2)
        st.success(f"✅ Entry saved! Emotion detected: **{emotion}**")

if not entries:
    st.info("No entries yet. Start by writing one!")
    st.stop()

# Dataframe
df = pd.DataFrame(entries)
df['datetime'] = pd.to_datetime(df['timestamp'])
df.sort_values('datetime', inplace=True)

# Filters
filtered_entries = entries
if keyword:
    filtered_entries = [e for e in filtered_entries if keyword.lower() in e['entry'].lower()]
if emotion_filter != "All":
    filtered_entries = [e for e in filtered_entries if e['emotion'] == emotion_filter]

# Personal insights
st.markdown("---")
st.header("📊 Personal Insights")
emotion_counts = df['emotion'].value_counts().reset_index()
emotion_counts.columns = ['emotion', 'count']
common_emotion = emotion_counts.iloc[0]['emotion']
st.markdown(f"🎯 **Most common emotion:** {common_emotion}")

# Happiest day
happiest_day = df[df['emotion']=='joy']['datetime'].dt.date.mode()
if len(happiest_day) > 0:
    st.markdown(f"😊 **Your happiest day was:** {happiest_day.iloc[0]}")

# Active time of day
df['hour'] = df['datetime'].dt.hour
active_hour = df['hour'].mode()
if len(active_hour) > 0:
    st.markdown(f"🕰 **You usually write around:** {active_hour.iloc[0]}:00")

# Weekly summary (this week)
this_week_start = pd.Timestamp(datetime.now()) - timedelta(days=7)
this_week_df = df[df['datetime'] >= this_week_start]
weekly_emotions = this_week_df['emotion'].value_counts().to_dict()
weekly_topics = " ".join([e['entry'] for e in entries if pd.Timestamp(e['timestamp']) >= this_week_start])
weekly_keywords = [kw[0] for kw in kw_model.extract_keywords(weekly_topics, top_n=5)]
st.markdown(
    f"🧠 **This week’s themes:** {', '.join(weekly_keywords)} — "
    f"Emotions felt: {weekly_emotions}"
)

# Mood heatmap (days vs emotion)
st.markdown("## 🧭 Mood Calendar Heatmap")
df['date'] = df['datetime'].dt.date
heat_df = df.groupby(['date','emotion']).size().reset_index(name='count')
heat_pivot = heat_df.pivot(index='date', columns='emotion', values='count').fillna(0)
fig_heat = px.imshow(
    heat_pivot.values,
    x=heat_pivot.columns,
    y=heat_pivot.index.astype(str),
    color_continuous_scale='RdYlBu',
    title='Emotions per day'
)
st.plotly_chart(fig_heat, use_container_width=True)

# Emotional journey timeline with date range filter
st.markdown("## 📈 Emotional Journey Timeline")
min_date, max_date = df['datetime'].min().date(), df['datetime'].max().date()

# Only show slider if we have a valid range
if min_date < max_date:
    date_range = st.slider(
        "📆 Filter date range",
        min_value=min_date, max_value=max_date,
        value=(min_date, max_date)
    )
    filtered_df = df[(df['datetime'].dt.date >= date_range[0]) & (df['datetime'].dt.date <= date_range[1])]
else:
    st.info(f"Only one entry on {min_date}. Showing all data.")
    filtered_df = df.copy()

# Proceed with emotion_num and the timeline plot
emotion_order = ["neutral","sadness","disgust","fear","anger","surprise","joy"]
filtered_df['emotion_num'] = filtered_df['emotion'].apply(
    lambda e: emotion_order.index(e) if e in emotion_order else -1
)
fig_line = px.line(
    filtered_df, x='datetime', y='emotion_num',
    title='Your Emotional Journey',
    markers=True
)
fig_line.update_yaxes(
    tickvals=list(range(len(emotion_order))),
    ticktext=emotion_order
)
st.plotly_chart(fig_line, use_container_width=True)

# Word Cloud
st.markdown("## ☁️ Word Cloud of Entries")
all_text = " ".join([e['entry'] for e in entries]).lower()
wc = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(all_text)
st.image(wc.to_image(), use_container_width=True)

# Past entries
st.markdown("## 📜 Past Entries")
if filtered_entries:
    for e in reversed(filtered_entries):
        color = "#D0F0C0" if e['emotion']=="joy" else "#FFE5E5"
        doc = nlp_ner(e['entry'])
        entities_text = ", ".join([f"{ent.text} ({ent.label_})" for ent in doc.ents])
        keywords = kw_model.extract_keywords(e['entry'], top_n=3)
        keywords_text = ", ".join([kw[0] for kw in keywords])
        st.markdown(
            f"""
            <div style='background-color:{color}; padding:10px; border-radius:8px; margin:10px 0;'>
                <b>{e['timestamp']} — {e['emotion']}</b><br>
                <i>{e['summary']}</i><br>
                <small>{e['entry']}</small><br>
                🔑 Keywords: {keywords_text}<br>
                🏷 Entities: {entities_text}<br>
                💭 AI Prompt: Would you like to explore this feeling of {e['emotion']} more?
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.info("No entries match your filters/search.")

# Export options
st.markdown("---")
st.header("💾 Export Options")

# Export as Markdown
md_text = "# Memory Lens Journal\n\n" + "\n\n".join(
    f"**{e['timestamp']}** — *{e['emotion']}*\n\n{e['entry']}\n\n_{e['summary']}_" for e in filtered_entries
)
st.download_button(
    label="📄 Download as Markdown",
    data=md_text,
    file_name="journal.md",
    mime="text/markdown"
)

# Export as PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font('Helvetica', size=12)
pdf.cell(0, 10, "Memory Lens Journal", ln=True)
for e in filtered_entries:
    title_text = f"{e['timestamp']} - {e['emotion']}"
    title_text = title_text.encode('latin-1', errors='replace').decode('latin-1')
    entry_text = e['entry'].encode('latin-1', errors='replace').decode('latin-1')
    pdf.cell(0, 10, title_text, ln=True)
    pdf.multi_cell(0, 10, entry_text)
    pdf.cell(0, 10, "", ln=True)
pdf_bytes = pdf.output(dest='S').encode('latin-1')
st.download_button(
    label="📄 Download as PDF",
    data=pdf_bytes,
    file_name="journal.pdf",
    mime="application/pdf"
)

# Save single entry as image
st.markdown("### 🖼 Save Entry as Image Card")
entry_to_save = st.selectbox(
    "Choose an entry to save as image:",
    options=[e['timestamp'] for e in entries]
)
selected = next(e for e in entries if e['timestamp']==entry_to_save)
image_card_text = f"{selected['timestamp']} — {selected['emotion']}\n\n{selected['summary']}\n\n{selected['entry']}"
image_card = Image.new('RGB', (800, 400), color='white')
draw = ImageDraw.Draw(image_card)
draw.text((10,10), image_card_text, fill='black')
buf = BytesIO()
image_card.save(buf, format='PNG')
buf.seek(0)
st.download_button(
    label="💾 Download Entry as Image",
    data=buf,
    file_name="entry.png",
    mime="image/png"
)
